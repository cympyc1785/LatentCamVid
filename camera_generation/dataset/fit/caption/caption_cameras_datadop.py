"""Vista4D eval 카메라(합성 target / recon 소스)에 **DataDoP 방식 그대로** 카메라 캡션을 단다.

WHY: GenDoP 은 DataDoP 캡션 분포로 학습됐다. 우리 카메라를 GenDoP 에 물리려면 캡션이
     "비슷한 말"이면 안 되고 **같은 파이프라인 산출물**이어야 한다. 그래서 문구를 새로 짜지 않고
     DataDoP 의 세 단계를 그대로 태운다:
       ① `segment_rigidbody_trajectories` (translation 27 패턴 × angular 7 패턴)
       ② outline → LLM → `Movement`            (`llm/cam+rotate.yaml` 프롬프트 그대로)
       ③ 영상 16프레임 4×4 그리드 + Movement → LLM → `Detailed`/`Concise Interaction`
          (`llm/relationship+image.json` 프롬프트 그대로)
     GenDoP 리포는 **0줄 수정**하고 `core.utils` / `processing.segmentation` 만 import 한다.

규약 세 가지 (전부 실측/코드 근거):
  * **좌표계**: DataDoP `_transforms_cleaning.json` 은 OpenGL c2w 다. 근거는 코드 두 곳 —
    `Dataset_DataDoP.convert_viser_poses_to_new_coordinate_system` 이 `matrix[:3,1:3] *= -1`
    을 걸고, `CAM_INDEX_TO_PATTERN` 이 +z=backward / +y=up / +x=right 로 라벨링한다.
    우리 `cam_c2w` 는 OpenCV 라 같은 flip 을 걸어서 넘긴다.
  * **프레임 수**: DataDoP 는 shot 길이와 무관하게 120 pose 로 리샘플하고 fps 는 30.0 으로 **고정**
    한다 (`pose_clean_normalize` 의 `range(120)`, `caption_cam+char.yaml` 의 `fps: 30.0`).
    우리 49프레임을 그대로 넣으면 smoothing window(15~) 가 궤적의 1/3 을 덮어 전부 1-segment 가
    된다. 그래서 같은 slerp(`sample_from_dense_cameras`)로 120 으로 늘린다.
  * **스케일**: 리스케일하지 않는다 (D=1). 실측 근거 — DataDoP 22,314 클립의 클립당 총이동
    median 0.1016 vs 우리 recon 72편 0.1664 로 같은 자릿수다. MonST3R 가 이미 shot 단위
    정규화를 해둔 게이지라 depth 로 또 나누면 산포만 커진다.

**한계 — zoom 은 캡션에 안 들어간다.** DataDoP 어휘는 translation+rotation 뿐이라
`zoom-out` 처럼 위치 고정 + focal 램프인 카메라는 "static" 으로 적힌다. focal 비는 캡션에 못
싣는 대신 `_tag.json` 의 `focal_ratio` 에 남기고 실행 끝에 몇 개가 그런지 세어 보고한다.

사용 예시:
    # 태깅만 (LLM 없이) — 방향 라벨이 preset 이름과 맞는지 먼저 본다
    conda run -n GenDoP python fit/caption/caption_cameras_datadop.py --sets cameras --no_llm

    # 전량 캡션 (로컬 Qwen3-VL)
    conda run -n GenDoP python fit/caption/caption_cameras_datadop.py --sets cameras recon
"""
from argparse import ArgumentParser
from glob import glob
from os import makedirs, path
import json
import sys

import numpy as np
import torch

VENDORED_GENDOP = "/data1/cympyc1785/LatentCamVid/camera_generation/models/GenDoP"
PIPELINE_GENDOP = "/data1/cympyc1785/pipeline/GenDoP"
# WHY argparse 전에 읽나: `processing.segmentation` 은 import 시점에 경로가 정해져야 한다.
# 두 리포는 같은 함수의 **하드코딩된 combine 단계 값**이 다르다 — vendored `min_chunk_size = 10`
# vs pipeline `= 12` (`segment_rigidbody_trajectories` 안에서 인자를 덮어쓴다). 인자로 못 바꾸니
# 리포 자체를 갈아끼워야 "pipeline 세팅대로"가 된다.
GENDOP_ROOT = VENDORED_GENDOP
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gendop_root" and _i + 1 < len(sys.argv):
        GENDOP_ROOT = sys.argv[_i + 1]
    elif _arg.startswith("--gendop_root="):
        GENDOP_ROOT = _arg.split("=", 1)[1]
CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
for _extra in (GENDOP_ROOT, path.join(GENDOP_ROOT, "dataset/scripts"), CINEMATRAJ_ROOT):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from core.utils import sample_from_dense_cameras                      # noqa: E402
from processing.segmentation import (                                 # noqa: E402
    ANG_INDEX_TO_PATTERN,
    CAM_INDEX_TO_PATTERN,
    find_consecutive_chunks,
    segment_rigidbody_trajectories,
)
try:    # CAMERABENCH 어휘표는 pipeline 리포에만 있다 (vendored 에는 아예 없음).
    from processing.segmentation import (                             # noqa: E402
        CAMERABENCH_ANG_INDEX_TO_PATTERN,
        CAMERABENCH_CAM_INDEX_TO_PATTERN,
    )
except ImportError:
    CAMERABENCH_ANG_INDEX_TO_PATTERN = CAMERABENCH_CAM_INDEX_TO_PATTERN = None

from lbm.vlm import VLMClient, DEFAULT_API_BASE, DEFAULT_MODEL        # noqa: E402

# DataDoP 하이퍼파라미터 — `configs/captioning/{caption_cam+char,cam/segment_cam}.yaml` 그대로.
# WHY 노브로 뺐나: fps 는 `compute_camera_dynamics` 에서 velocity 에 그대로 곱해지는 눈금이다
# (`segmentation.py:92  t_velocities = fps * velocities[:, :3, 3]`). fps 를 30 -> 10 으로
# 내리면 같은 궤적의 속도가 1/3 이 되어 static/moving 임계 3종이 통째로 어긋난다. 기본값은
# DataDoP 원본 그대로 두고 CLI 로만 바꾼다 (기본 인자만 주면 이전과 bit-identical).
SEG_DEFAULTS = dict(cam_static_threshold=0.02, cam_diff_threshold=0.4,
                    angular_static_threshold=0.005, fps=30.0,
                    smoothing_window_size=18, min_chunk_size=10)
LLM_CFG = path.join(GENDOP_ROOT, "dataset/scripts/configs/captioning/llm")


def load_prompts():
    """DataDoP 의 두 프롬프트를 원본 파일에서 읽는다 (하드코딩하면 원본과 어긋난다)."""
    import yaml

    with open(path.join(LLM_CFG, "cam+rotate.yaml"), encoding="utf-8") as file:
        movement = yaml.safe_load(file)
    with open(path.join(LLM_CFG, "relationship+image.json"), encoding="utf-8") as file:
        relationship = json.load(file)
    return movement, relationship


# ------------------------------------------------------------------------------------- #
# 궤적 -> outline

def opencv_c2w_to_datadop(cam_c2w):
    """OpenCV c2w -> DataDoP(OpenGL) c2w. `convert_viser_poses_...` 의 `[:3,1:3] *= -1`."""
    poses = np.asarray(cam_c2w, dtype=np.float64).copy()
    poses[:, :3, 1:3] *= -1.0
    return poses


def resample_poses(poses, num_poses):
    """DataDoP `pose_clean_normalize` 와 같은 slerp 리샘플 (t = i/num_poses, 끝은 안 닿는다).

    `num_poses <= 0` 이면 리샘플하지 않고 원본 프레임을 그대로 쓴다 (Vista 49프레임 경로).
    """
    if num_poses <= 0:
        return np.asarray(poses, dtype=np.float64)
    dense = torch.tensor(poses[:, :3, :].reshape(1, -1, 12), dtype=torch.float32)
    rows = [sample_from_dense_cameras(dense, torch.full((1, 1), i / num_poses))[0]
            for i in range(num_poses)]
    out = np.tile(np.eye(4), (num_poses, 1, 1))
    out[:, :3, :] = torch.cat(rows, dim=0).numpy().reshape(num_poses, 3, 4)
    return out


def build_outline(segments, shuffle_taxonomy=False):
    """`processing/captioning.py:caption_trajectories` 의 `traj_description` 과 같은 문자열.

    `shuffle_taxonomy` 는 pipeline config `caption_cam+char.yaml:21` 의 그 키다 — True 면 chunk
    마다 DataDoP 어휘표와 CAMERABENCH 어휘표 중 하나를 `random.choice` 로 골라 라벨을 쓴다
    (`captioning.py:147`). **기본 False = pipeline 설정값**이고, 그 경우 이 함수는 어휘표 선택
    분기 자체를 타지 않아 예전 동작과 bit-identical 이다.
    """
    chunks = find_consecutive_chunks(list(segments))
    lines, end = [], 0
    for index, start, end in chunks:
        cam_table, ang_table = CAM_INDEX_TO_PATTERN, ANG_INDEX_TO_PATTERN
        if shuffle_taxonomy:
            import random
            cam_table = random.choice([CAM_INDEX_TO_PATTERN, CAMERABENCH_CAM_INDEX_TO_PATTERN])
            ang_table = random.choice([ANG_INDEX_TO_PATTERN, CAMERABENCH_ANG_INDEX_TO_PATTERN])
        move = cam_table[index // 7]
        angular = ang_table[index % 7]
        text = f"Between frames {start} and {end}: {move}"
        lines.append(text if angular == "static" else f"{text} + {angular}")
    outline = (f"\n\nOutline: Total frames {end + 1}. "
               + "\n[Camera motion] " + "; ".join(lines) + ". ")
    return outline, [(int(i), int(s), int(e)) for i, s, e in chunks]


def tag_trajectory(cam_c2w, num_poses, seg_kwargs=None, shuffle_taxonomy=False, convert=True):
    """`convert=False` 는 입력이 **이미 DataDoP(OpenGL) c2w** 일 때 쓴다.

    DataDoP 원본 `_transforms_cleaning.json` 이 그렇다 — 거기에 `[:3,1:3]*=-1` 을 또 걸면
    y/z 가 되돌아가 up/forward 부호가 뒤집히고, 라벨이 조용히 반대로 나온다. 기본값 `True` 는
    우리 npz(OpenCV c2w) 경로라 예전 동작과 bit-identical 이다.
    """
    seg_kwargs = dict(SEG_DEFAULTS) if seg_kwargs is None else dict(seg_kwargs)
    poses = np.asarray(cam_c2w, dtype=np.float64) if not convert \
        else opencv_c2w_to_datadop(cam_c2w)
    poses = resample_poses(poses, num_poses)
    segments = segment_rigidbody_trajectories(torch.tensor(poses, dtype=torch.float64),
                                              **seg_kwargs)
    outline, chunks = build_outline(segments, shuffle_taxonomy)
    rel = np.linalg.inv(poses[:-1]) @ poses[1:]
    step = np.linalg.norm(rel[:, :3, 3], axis=1)
    return {
        "outline": outline,
        "chunks": [{"index": i, "start": s, "end": e,
                    "move": CAM_INDEX_TO_PATTERN[i // 7],
                    "angular": ANG_INDEX_TO_PATTERN[i % 7]} for i, s, e in chunks],
        "segments": [int(x) for x in segments],
        "num_poses": int(poses.shape[0]),
        "seg_kwargs": seg_kwargs,
        "shuffle_taxonomy": bool(shuffle_taxonomy),
        "step_median": float(np.median(step)),
        "total_translation": float(np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3])),
    }


# ------------------------------------------------------------------------------------- #
# 영상 -> 4x4 그리드

def build_grid(video_path, out_path, num_tiles=16, tile_width=320):
    """영상에서 균등 16프레임을 뽑아 4×4 로 붙인다 (`relationship+image.json` 이 가정하는 배치).

    `video_path` 가 **디렉토리**면 그 안의 png/jpg 를 파일명 순으로 프레임으로 본다 —
    latentcam 코퍼스는 mp4 가 없고 `images_4/00000.png..00048.png` 만 있다.
    """
    import cv2

    frames = []
    if path.isdir(video_path):
        files = sorted(glob(path.join(video_path, "*.png"))
                       + glob(path.join(video_path, "*.jpg")))
        for file_path in files:
            image = cv2.imread(file_path)
            if image is not None:
                frames.append(image)
    else:
        capture = cv2.VideoCapture(video_path)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
    if len(frames) < num_tiles:
        raise ValueError(f"{video_path}: 프레임 {len(frames)} < {num_tiles}")

    picks = np.linspace(0, len(frames) - 1, num_tiles).round().astype(int)
    height = int(round(tile_width * frames[0].shape[0] / frames[0].shape[1]))
    tiles = [cv2.resize(frames[i], (tile_width, height)) for i in picks]
    side = int(np.sqrt(num_tiles))
    grid = np.concatenate([np.concatenate(tiles[r * side:(r + 1) * side], axis=1)
                           for r in range(side)], axis=0)
    makedirs(path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path, len(frames)


# ------------------------------------------------------------------------------------- #
# LLM 두 번

def ask_movement(client, prompts, outline):
    """DataDoP `caption_trajectories` 와 같은 프롬프트 조립. 숫자가 섞이면 되묻는다."""
    prompt = (f"{prompts['context']}\n{prompts['instruction']}\n{prompts['constraint']}"
              f"{prompts['demonstration']}{outline}\nDescription: ")
    for _ in range(3):
        text, _meta = client.chat(prompt, label="movement")
        text = text.strip().split("\n")[0].strip()
        # 원본 `single_test` 도 숫자가 들어가면 실패로 보고 다시 부른다 (프레임 인덱스 누출 방지).
        if text and not any(character.isdigit() for character in text):
            return text
    return text


def ask_interaction(client, prompts, grid_path, movement):
    """4×4 그리드 + Movement -> `**Detailed**: ...` / `**Concise**: ...` 두 줄."""
    prompt = (prompts["context"] + prompts["instruction"] + prompts["constraint"]
              + prompts["format"] + "\n\nMovement: " + movement)
    detailed = concise = ""
    for _ in range(3):
        text, _meta = client.chat(prompt, images=[grid_path], label="interaction")
        for line in text.splitlines():
            stripped = line.strip().lstrip("*").strip()
            lowered = stripped.lower()
            if lowered.startswith("detailed"):
                detailed = stripped.split(":", 1)[-1].strip().strip("*").strip()
            elif lowered.startswith("concise"):
                concise = stripped.split(":", 1)[-1].strip().strip("*").strip()
        if detailed and concise:
            break
    return detailed, concise


# ------------------------------------------------------------------------------------- #

def load_datadop_json(json_path):
    """DataDoP `<shot>_transforms_cleaning.json` -> npz 와 같은 모양의 dict.

    `frames[].transform_matrix` 는 **이미 OpenGL c2w** 라 `[:3,1:3]*=-1` 을 걸면 안 된다
    (`tag_trajectory(convert=False)` 로 넘긴다). 그래서 키 이름을 `cam_gl_c2w` 로 따로 둔다 —
    `load_entry_cameras` 가 `extrinsics` 만 w2c 로 보고 뒤집기 때문에 이름이 곧 규약 표시다.
    intrinsics 는 shot 전체가 상수(`fl_x`)라 프레임 수만큼 복제한다.
    """
    with open(json_path, encoding="utf-8") as file:
        meta = json.load(file)
    frames = meta["frames"]
    poses = np.asarray([f["transform_matrix"] for f in frames], dtype=np.float64)
    K = np.tile(np.eye(3), (len(frames), 1, 1))
    K[:, 0, 0], K[:, 1, 1] = meta["fl_x"], meta["fl_y"]
    K[:, 0, 2], K[:, 1, 2] = meta["cx"], meta["cy"]
    return {"cam_gl_c2w": poses, "intrinsics": K}


def collect_entries(root, sets, latentcam_root=None, latentcam_split=None,
                    datadop_root=None, datadop_valid=None, datadop_out=None,
                    datadop_limit=0, datadop_seed=0):
    """(set, scene, name, npz_path, cam_key, video_path, out_dir, entry_index) 목록.

    `entry_index` 는 latentcam 전용 — 한 scene 의 npz 하나에 N개 target 이 쌓여 있어서
    행 인덱스가 있어야 한 엔트리를 집는다. 기존 두 set 은 npz 1개 = 카메라 1개라 None.
    """
    entries = []
    if "datadop" in sets:
        # DataDoP GT 원본. `DataDoP_valid.txt` 한 줄 = `<scene>/<shot>`, 카메라는
        # `<scene>/<shot>_transforms_cleaning.json` 의 120 pose 다 (npz 가 아니라 json).
        # WHY 서브샘플: valid 28,971 편 전부를 태깅할 이유가 없다 — 우리 뱅크(1e4 규모)와
        # 대조하는 거라 같은 자릿수면 충분하고, seed 고정 셔플이라 재현된다.
        with open(datadop_valid, encoding="utf-8") as file:
            shots = [line.strip() for line in file if line.strip()]
        shots = [s for s in shots
                 if path.exists(path.join(datadop_root,
                                          f"{s}_transforms_cleaning.json"))]
        if datadop_limit and datadop_limit < len(shots):
            rng = np.random.default_rng(datadop_seed)
            shots = [shots[i] for i in sorted(rng.permutation(len(shots))[:datadop_limit])]
        for shot in shots:
            scene, name = shot.split("/")
            entries.append(("datadop", scene, name,
                            path.join(datadop_root, f"{shot}_transforms_cleaning.json"),
                            "cam_gl_c2w", None,
                            path.join(datadop_out, scene), None))
    if "cameras" in sets:
        for npz_path in sorted(glob(path.join(root, "cameras", "*", "*.npz"))):
            scene = path.basename(path.dirname(npz_path))
            name = path.basename(npz_path)[:-4]
            videos = sorted(glob(path.join(root, "gen", scene, name, "video_seed=*.mp4")))
            entries.append(("cameras", scene, name, npz_path, "cam_c2w",
                            videos[0] if videos else None,
                            path.join(root, "cameras", scene, "captions"), None))
    if "recon" in sets:
        for npz_path in sorted(glob(path.join(root, "recon_and_seg", "*", "cameras.npz"))):
            scene = path.basename(path.dirname(npz_path))
            video = path.join(root, "recon_and_seg", scene, "video.mp4")
            entries.append(("recon", scene, "source", npz_path, "cam_c2w",
                            video if path.exists(video) else None,
                            path.join(root, "recon_and_seg", scene, "captions"), None))
    if "latentcam" in sets:
        # split 파일 한 줄 = `vista4d/<scene>/<entry_index>`. 카메라는 scene 마다 하나의
        # `da3/target_poses.npz` 에 (N,49,4,4) 로 쌓여 있고 그 행이 곧 entry_index 다.
        with open(latentcam_split, encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
        for line in lines:
            dataset, scene, index = line.split("/")
            scene_dir = path.join(latentcam_root, dataset, scene)
            entries.append(("latentcam", scene, index,
                            path.join(scene_dir, "da3", "target_poses.npz"), "extrinsics",
                            path.join(scene_dir, "images_4"),
                            path.join(scene_dir, "da3", "captions_gendop"), int(index)))
    return entries


def load_entry_cameras(data, cam_key, entry_index):
    """npz -> (OpenCV c2w (F,4,4), focal (F,) or None).

    latentcam(`extrinsics`) 는 **OpenCV w2c** 다 — 실측: `inv(M) @ diag(1,-1,-1,1)` 가
    eval 쪽 레퍼런스(OpenGL c2w)와 max|diff| 2.71e-07 로 일치했고, `M` / `M@GL2CV` /
    `inv(M)` 은 각각 3.64 / 3.64 / 2.0 으로 어긋났다. 그래서 여기서 뒤집어 c2w 로 만든다.
    기존 두 set 의 `cam_c2w` 는 이미 OpenCV c2w 라 그대로 통과한다.
    """
    poses = data[cam_key]
    intrinsics = data["intrinsics"] if "intrinsics" in data else None
    if entry_index is not None:                     # (N,F,4,4) / (N,F,3,3) 에서 한 행
        poses = poses[entry_index]
        intrinsics = None if intrinsics is None else intrinsics[entry_index]
    poses = np.asarray(poses, dtype=np.float64)
    if cam_key == "extrinsics":                     # w2c -> c2w
        poses = np.linalg.inv(poses)
    focal = None
    if intrinsics is not None:
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        # (F,3,3) 이면 fx = [:,0,0], 기존 npz 처럼 (F,3) 로 눌려 있으면 [:,0].
        focal = intrinsics[:, 0, 0] if intrinsics.ndim == 3 else intrinsics[:, 0]
    return poses, focal


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data1/cympyc1785/LatentCamVid/DATA/"
                                          "Vista4D-Eval-Data/eval_data")
    parser.add_argument("--sets", nargs="+", default=["cameras", "recon"],
                        choices=["cameras", "recon", "latentcam", "datadop"])
    # datadop = DataDoP GT 원본 (분포 대조용). 우리 뱅크와 **같은 seg_kwargs / 같은 gendop_root**
    # 로 돌려야 비교가 성립한다 — 임계가 다르면 차이가 코퍼스가 아니라 노브에서 나온다.
    parser.add_argument("--datadop_root",
                        default="/data1/cympyc1785/data/DataDoP/DataDoP_with_scene")
    parser.add_argument("--datadop_valid",
                        default="/data1/cympyc1785/data/DataDoP/DataDoP_valid.txt")
    parser.add_argument("--datadop_out",
                        default=path.join(CINEMATRAJ_ROOT, "out_captions", "datadop"))
    parser.add_argument("--datadop_limit", type=int, default=0)   # 0 = valid 전량
    parser.add_argument("--datadop_seed", type=int, default=0)
    # latentcam = LBM-Lite 로 합성한 target 카메라 코퍼스 (scene 마다 npz 하나에 N개가 쌓여 있다)
    parser.add_argument("--latentcam_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    parser.add_argument("--latentcam_split", default=None)   # None = <root>/seg_list_vista4d_test.txt
    parser.add_argument("--num_poses", type=int, default=120)   # DataDoP 고정값, 0 = 리샘플 안 함
    # 분절기 노브 5종 — 기본값은 DataDoP 원본 (`SEG_DEFAULTS`). Vista 카메라용으로 재조정한다.
    parser.add_argument("--fps", type=float, default=SEG_DEFAULTS["fps"])
    parser.add_argument("--static_threshold", type=float,
                        default=SEG_DEFAULTS["cam_static_threshold"])
    parser.add_argument("--diff_threshold", type=float,
                        default=SEG_DEFAULTS["cam_diff_threshold"])
    parser.add_argument("--angular_static_threshold", type=float,
                        default=SEG_DEFAULTS["angular_static_threshold"])
    parser.add_argument("--smoothing_window_size", type=int,
                        default=SEG_DEFAULTS["smoothing_window_size"])
    parser.add_argument("--min_chunk_size", type=int, default=SEG_DEFAULTS["min_chunk_size"])
    # pipeline `caption_cam+char.yaml:21` 의 그 키. False 가 pipeline 값이자 우리 기존 동작.
    parser.add_argument("--shuffle_taxonomy", dest="shuffle_taxonomy", action="store_true")
    parser.add_argument("--no_shuffle_taxonomy", dest="shuffle_taxonomy", action="store_false")
    parser.set_defaults(shuffle_taxonomy=False)
    parser.add_argument("--only", nargs="+", default=None)      # "scene/name" 몇 개만
    # 분절기를 어느 GenDoP 리포에서 import 할지 (import 시점에 이미 소비됐다, 위 pre-scan 참조).
    parser.add_argument("--gendop_root", default=VENDORED_GENDOP)
    # WHY: DataDoP 원본 세팅으로 만든 `captions/` 를 덮지 않으려고 출력 폴더를 나눈다.
    parser.add_argument("--out_subdir", default="captions")
    parser.add_argument("--no_llm", action="store_true")        # 태깅만 (outline 까지)
    parser.add_argument("--api_base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    args = parser.parse_args()

    if args.shuffle_taxonomy and CAMERABENCH_CAM_INDEX_TO_PATTERN is None:
        parser.error(f"--shuffle_taxonomy 는 CAMERABENCH 어휘표가 있는 리포에서만 된다 "
                     f"(지금 --gendop_root={GENDOP_ROOT} 에는 없다)")

    seg_kwargs = dict(cam_static_threshold=args.static_threshold,
                      cam_diff_threshold=args.diff_threshold,
                      angular_static_threshold=args.angular_static_threshold,
                      fps=args.fps, smoothing_window_size=args.smoothing_window_size,
                      min_chunk_size=args.min_chunk_size)

    latentcam_split = args.latentcam_split or path.join(args.latentcam_root,
                                                        "seg_list_vista4d_test.txt")
    entries = collect_entries(args.root, args.sets, args.latentcam_root, latentcam_split,
                              args.datadop_root, args.datadop_valid, args.datadop_out,
                              args.datadop_limit, args.datadop_seed)
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if f"{e[1]}/{e[2]}" in wanted or e[1] in wanted]
    entries = [e for i, e in enumerate(entries) if i % args.num_shards == args.shard_id]
    if args.out_subdir != "captions":       # 기본 폴더가 아니면 out_dir 의 끝만 갈아끼운다
        # datadop 은 out_dir 가 `<datadop_out>/<scene>` 이라 끝을 갈아끼우면 scene 이 날아간다.
        # 그쪽 출력 분리는 `--datadop_out` 이 담당하므로 여기서 건너뛴다.
        entries = [e if e[0] == "datadop"
                   else e[:6] + (path.join(path.dirname(e[6]), args.out_subdir),) + e[7:]
                   for e in entries]
    prompts_movement, prompts_relationship = load_prompts()
    client = None if args.no_llm else VLMClient(api_base=args.api_base, model=args.model)

    done, skipped, no_video, zoomish = 0, 0, [], []
    npz_cache = {}      # latentcam 은 scene 당 npz 하나를 수백 엔트리가 공유한다
    for kind, scene, name, npz_path, cam_key, video_path, out_dir, entry_index in entries:
        caption_path = path.join(out_dir, f"{name}_caption.json")
        # `--no_llm` 은 caption 을 안 쓰므로 caption_path 로 건너뛰면 매번 전량 재태깅이 된다.
        # 태깅만 도는 경로에서는 최종 산출물이 `_tag.json` 이라 그쪽을 본다.
        done_path = path.join(out_dir, f"{name}_tag.json") if args.no_llm else caption_path
        if (not args.overwrite) and path.exists(done_path):
            skipped += 1
            continue

        if npz_path not in npz_cache:
            npz_cache = {npz_path: (load_datadop_json(npz_path) if kind == "datadop"
                                    else dict(np.load(npz_path)))}   # scene 넘어가면 통째 교체
        data = npz_cache[npz_path]
        cam_c2w, focal = load_entry_cameras(data, cam_key, entry_index)
        # datadop 원본은 이미 OpenGL c2w -> flip 금지 (`tag_trajectory` docstring 참조)
        tag = tag_trajectory(cam_c2w, args.num_poses, seg_kwargs, args.shuffle_taxonomy,
                             convert=(kind != "datadop"))
        tag.update(kind=kind, scene=scene, name=name, cam_key=cam_key, gendop_root=GENDOP_ROOT,
                   source_npz=(npz_path if kind in ("latentcam", "datadop")  # root 밖이라 relpath 불가
                               else path.relpath(npz_path, args.root)),
                   entry_index=entry_index,
                   num_source_frames=int(cam_c2w.shape[0]),
                   convention=("DataDoP OpenGL c2w (원본 그대로, flip 없음), 스케일 D=1"
                               if kind == "datadop" else
                               "DataDoP OpenGL c2w (OpenCV c2w 에 [:3,1:3]*=-1), 스케일 D=1"))
        if entry_index is not None:     # 어느 anchor/preset/rung 인지 추적 가능하게
            for key in ("keys", "variant_id"):
                if key in data:
                    tag[key] = str(data[key][entry_index])
        if focal is not None:
            tag["focal_ratio"] = float(focal[-1] / focal[0])
            if abs(np.log(tag["focal_ratio"])) > np.log(1.1):
                zoomish.append(f"{scene}/{name} f×{tag['focal_ratio']:.2f}")

        makedirs(out_dir, exist_ok=True)
        with open(path.join(out_dir, f"{name}_tag.json"), "w", encoding="utf-8") as file:
            json.dump(tag, file, ensure_ascii=False, indent=2)

        if args.no_llm:
            print(f"[tag ] {kind:<8}{scene}/{name}\n       {tag['outline'].strip()}")
            done += 1
            continue

        if video_path is None:
            no_video.append(f"{scene}/{name}")
            continue
        # latentcam 은 scene 전체가 같은 소스 프레임을 쓴다 -> 그리드를 scene 당 하나만 만든다.
        grid_path = path.join(out_dir, "_scene_grid.png" if entry_index is not None
                              else f"{name}_grid.png")
        if not path.exists(grid_path):
            build_grid(video_path, grid_path)

        movement = ask_movement(client, prompts_movement, tag["outline"])
        detailed, concise = ask_interaction(client, prompts_relationship,
                                            grid_path, movement)
        with open(caption_path, "w", encoding="utf-8") as file:
            json.dump({"Movement": movement, "Detailed Interaction": detailed,
                       "Concise Interaction": concise}, file,
                      ensure_ascii=False, indent=4)
        done += 1
        print(f"[cap ] {kind:<8}{scene}/{name}\n       {movement}")

    print()
    print(f"{'seg_kwargs':<20}{seg_kwargs}")
    print(f"{'num_poses':<20}{args.num_poses if args.num_poses > 0 else 'native (리샘플 없음)'}")
    print(f"{'entries':<20}{len(entries)}")
    print(f"{'written':<20}{done}")
    print(f"{'skipped (exists)':<20}{skipped}")
    print(f"{'no video':<20}{len(no_video)}  {no_video[:5]}")
    print(f"{'focal ramp >1.1x':<20}{len(zoomish)}  {zoomish[:5]}")
    print(f"{'note':<20}zoom 은 DataDoP 어휘에 없다 — focal_ratio 는 _tag.json 에만 있다")


if __name__ == "__main__":
    main()
