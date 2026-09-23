"""소스 영상 1편(RGBD + 카메라 + 마스크)을 4D point cloud 로 올리고 캐시한다.

왜 이게 필요한가: look-before-move 의 전제는 "후보 카메라를 **먼저 렌더해서** VLM 에게 보여준다"
는 것이다. 그 렌더러가 Vista4D 의 `utils/point_cloud/point_cloud.py` 이고, 그게 먹는 입력이 바로
`unproject()` 가 뱉는 (colors, points_world, visible, indices) 4-tuple 이다. 후보를 수십~수백 개
찍어보는 동안 매번 unproject 를 다시 돌 이유가 없으므로 여기서 한 번 만들고 npz 로 캐시한다.

`visible (n, f)` 가 이 자료구조의 핵심이다 — 정적 점은 전 프레임에서 True, 동적 점은 자기 프레임
하나에서만 True 라 `visible.sum(1) == 1` 이 동적 점 판별식이 된다 (`point_cloud.py:281` 이 쓰는
그 식). 즉 3D 가 아니라 4D 다.

입력  : `<eval_data>/eval_data/recon_and_seg/<video>/` (video.mp4 / depths / cameras.npz /
        dynamic_mask / sky_mask)
        `<eval_data>/eval_data/seg_instances/<video>/` (선택: SAM3 per-instance track)
출력  : `<out>/<video>/cloud.npz`  (format `lbm_cloud_v1`)

예시 (env vista4d 필요):
    CUDA_VISIBLE_DEVICES=0 python -m lbm.cloud --video camel
    CUDA_VISIBLE_DEVICES=0 python -m lbm.cloud --video camel --subject_track 1 --no_skip_done
"""
import sys
from os import makedirs, path

import numpy as np
import torch

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "video_generation", "models", "Vista4D"))
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"

CLOUD_FORMAT = "lbm_cloud_v1"


def import_vista4d(vista4d_root: str):
    """Vista4D 는 패키지가 아니라 `utils.*` 로 절대 import 하는 스크립트 트리라 sys.path 에 얹는다."""
    if vista4d_root not in sys.path:
        sys.path.insert(0, vista4d_root)
    from utils.media import intrinsics_to_K, load_recon_and_seg
    from utils.point_cloud.point_cloud import render, render_frame, unproject
    from utils.point_cloud.preprocess import SKY_DEPTH, preprocess_scene
    return {
        "load_recon_and_seg": load_recon_and_seg, "intrinsics_to_K": intrinsics_to_K,
        "unproject": unproject, "render_frame": render_frame, "render": render,
        "preprocess_scene": preprocess_scene, "SKY_DEPTH": SKY_DEPTH,
    }


# S 의 정의는 `scene_graph/scale.py` 한 곳뿐이다. 예전엔 여기에 복사본이 있었는데, 정의가 두
# 군데면 한쪽만 고쳐도 아무 에러 없이 게이지가 갈린다 (2026-09-02 정의 변경 때 실제로 위험했다).
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
from scene_graph.scale import SCALE_MODES, scene_scale                          # noqa: E402,F401


def parallax_ratio(cam_c2w: np.ndarray, z_med: float):
    """max_ij‖C_i − C_j‖ / z_med. 0.02 근처면 single-view depth shell 이라 보면 된다."""
    centers = cam_c2w[:, :3, 3]
    spread = float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())
    return spread / max(z_med, 1e-9)


def preprocess_recon(recon: dict, vista4d, depth_outliers: str = "gaussian", ignore_sky_mask: bool = False):
    """Vista4D 표준 전처리(`preprocess_scene`)를 unproject 앞에 끼운다. **이걸 건너뛰면 안 된다.**

    처음엔 `load_recon_and_seg` 결과를 `unproject` 에 곧장 넣었다가 소스 pose 재렌더 PSNR 이
    20 dB 에서 멈췄다. 배경은 완벽했고 동적 물체 주변만 반투명 빗살로 뭉개졌다. 원인은 규약이
    아니라 **static 점의 시간 지속성**이다: `static_mask = ~dynamic_mask` 는 마스크 경계에서
    프레임당 0.04% 쯤 물체 표면을 흘리는데, static 점은 `visible` 이 전 프레임 True 라
    (`point_cloud.py:63`) 그 누수가 49프레임어치 한꺼번에 렌더된다. 게다가 유령은 다른 프레임의
    물체 깊이에 앉아 있어 진짜 물체와 z 가 거의 같고, `render_frame` 의 z-tolerance 가 log1p
    공간 0.02 라 가려지는 대신 **블렌딩**된다. 그래서 빗살이 된다.

    `preprocess_scene` 이 정확히 이걸 막는다: static mask 를 erode(6px x3) 해서 경계 누수를
    잘라내고, depth outlier 를 지우고, sky 를 static 에 합치며 depth 를 SKY_DEPTH 로 박는다.
    해상도는 native 를 넘겨 resize 를 no-op 으로 만든다 (`media.py:resize_2d` 가 동일 해상도면
    바로 반환).

    sky 처리의 부수효과 하나: sky 가 static 점이 되므로 렌더에서 hole 이 아니라 먼 배경으로
    채워진다. G2 coverage 가 하늘 때문에 깎이지 않게 되는 건 의도된 것이다 (저자 기본 동작).
    """
    height, width = recon["video"].shape[1:3]
    processed = vista4d["preprocess_scene"](
        recon, indices=np.arange(recon["video"].shape[0]), height=height, width=width,
        depth_outliers=depth_outliers, ignore_sky_mask=ignore_sky_mask,
    )
    processed["fps"] = recon["fps"]
    return processed


_LINALG_WARMED = set()


def warm_linalg(device: str = "cuda"):
    """점군을 올리기 **전에** cuSOLVER 핸들을 만들어 둔다. 두 번째부터는 아무 일도 안 한다.

    왜 필요한가: Vista4D `render_frame()` 첫 줄이 `cam_c2w.inverse()` 인데, 4x4 짜리인데도
    torch 는 cuSOLVER 경로를 타고 첫 호출에서 `cusolverDnCreate` 를 부른다. 그 핸들은 torch
    캐싱 할당자 **바깥에서** cudaMalloc 을 하므로, 점군이 이미 카드를 채운 뒤에 만들려 하면
    `CUSOLVER_STATUS_INTERNAL_ERROR` 로 죽는다 — 4x4 역행렬이 메모리 부족으로 터지는 셈이라
    로그만 보면 원인이 안 보인다 (D181 파일럿 첫 팬아웃에서 3샤드가 전부 여기서 죽었다).
    핸들은 프로세스·디바이스 단위로 캐시되므로 한 번 미리 만들어 두면 그 뒤로는 안전하다.

    비용은 핸들 생성 한 번뿐이고 결과값은 안 쓴다. 실패해도 그냥 넘어간다 — 여기서 못 만들면
    어차피 뒤에서 같은 이유로 죽을 것이고, 이 함수가 새로운 실패 지점이 되면 안 된다.
    """
    if device in _LINALG_WARMED or not str(device).startswith("cuda"):
        return
    _LINALG_WARMED.add(device)
    try:
        torch.eye(4, device=device).inverse()
    except RuntimeError as err:  # 카드가 이미 꽉 찼다 — 여기서 죽이지는 않는다
        print(f"  [warm_linalg] cuSOLVER 예열 실패 (무시하고 진행): {err}", flush=True)


def build_cloud(recon: dict, vista4d, device: str = "cuda", dtype=torch.float32):
    """recon_and_seg dict → (colors, points_world, visible, indices) torch tensor 들.

    static_mask 를 넘겨야 정적 점이 전 프레임 visible 로 잡힌다 (temporal persistence). 안 넘기면
    모든 점이 자기 프레임에서만 보이게 되어 후보 렌더가 프레임 하나 분량의 점만 쓰고 텅 빈다.
    """
    K = vista4d["intrinsics_to_K"](recon["intrinsics"])
    warm_linalg(device)  # 점군 업로드 전에 cuSOLVER 핸들 확보 (warm_linalg docstring 참조)

    to = lambda a, d=dtype: torch.as_tensor(a, device=device).to(d)
    colors, points_world, visible, indices = vista4d["unproject"](
        video=torch.as_tensor(recon["video"], device=device),  # f h w 3, uint8 그대로
        depths=to(recon["depths"]),
        cam_c2w=to(recon["cam_c2w"]),
        K=to(K),
        dynamic_mask=to(recon["dynamic_mask"], torch.bool),
        static_mask=to(recon["static_mask"], torch.bool),
    )
    return colors, points_world, visible, indices, K


def subject_point_mask(indices: torch.Tensor, track_masks: np.ndarray):
    """`indices` (n,3)=[f,h,w] 를 (f,h,w) 마스크로 조회해 subject 점 인덱스를 만든다.

    track_masks 는 (f, h, w) bool — SAM3 track 하나의 per-frame 마스크. unproject 가 픽셀 순서를
    보존하지 않으므로(mask_flat 으로 걸러냄) 좌표로 되짚는 이 방법 말고는 대응이 안 붙는다.
    """
    lut = torch.as_tensor(track_masks, device=indices.device)  # f h w, bool
    f, h, w = indices[:, 0].long(), indices[:, 1].long(), indices[:, 2].long()
    return lut[f, h, w]


# D188-b. 이 가드가 터졌다는 것을 **로그 한 줄로** 알아볼 수 있게 하는 토큰. `run_bank.py` 의
# tau 단계가 이 토큰을 보고 크래시가 아니라 `skipped.json(empty_dynamic_mask)` 로 처리한다 —
# 안 그러면 이 씬들이 라운드마다 다시 ready 에 들어와 영원히 재시도된다 (실측: 10,346편 중 45편).
# 한국어 문장을 매칭하면 문구를 다듬는 순간 조용히 안 잡히므로 토큰을 따로 둔다.
EMPTY_DYNMASK_TOKEN = "[EMPTY_DYNAMIC_MASK]"


def assert_dynamic_mask_nonempty(dyn_frac: float, video: str, eval_data: str):
    """§empty_dynmask (D176, 2026-09-10) 가드.

    `dynpose_ingest.py:177` 의 recon 단계는 `seg_keywords=[]` 라 **all-zero placeholder**
    dynamic_mask 를 png 로 써 두고, 나중에 `dynmask` 단계가 SAM3 합집합으로 덮는다. 그 사이에
    cloud 를 구우면 `load_recon_and_seg` 가 `static_mask = ~dynamic_mask` 로 **전 픽셀을 정적**으로
    만들고, 정적 점은 `visible` 이 전 프레임 True 라 동적 물체가 49개 사본으로 잔류한다 — 에러
    없이, `visible.sum(1)==1` 이 정확히 0 으로. `load_masks` 는 png 가 **존재하는지**만 보므로
    (media.py:148) 새까만 png 는 통과한다. 실측 (dynpose 표본 120편): 47편(39%)이 이 상태였고
    `hole_fraction` med 0.199 vs 0.279, `subject_visible_frac<0.6` 36.8% vs 25.5% 로 편향됐다.
    bake 시각으로는 못 가른다 (정상인 씬의 cloud 가 오염된 씬보다 오래된 경우가 있다).

    함수로 뽑은 이유: npz 경로(`main()`)와 in-memory 경로(`cloud_from_recon`)가 **같은 판정**을
    써야 한다. 가드가 복사되면 한쪽만 고쳐도 조용히 빠지는데, 이 가드가 잡는 사고는 애초에 에러
    없이 통과하는 종류다.
    """
    if dyn_frac != 0.0:
        return
    seg_npz = path.join(eval_data, "eval_data", "seg_instances", video, "masks.npz")
    raise AssertionError(
        f"{EMPTY_DYNMASK_TOKEN} "
        f"{video}: dynamic_mask 가 전부 0 이다 — cloud 를 구우면 전 픽셀이 정적이 되어 "
        f"동적 물체가 49개 사본으로 남는다.\n"
        f"  seg_instances masks.npz {'있음' if path.isfile(seg_npz) else '없음'}: {seg_npz}\n"
        f"  있으면 `fit/ingest/dynpose_dynamic_mask_from_seg.py` (dynmask 단계) 를 먼저 돌릴 것.\n"
        f"  정말 동적 물체가 없는 씬이면 `--allow_empty_dynamic_mask` 로 통과시킨다.")


def cloud_from_recon(recon: dict, vista4d, *, video: str, eval_data: str,
                     S: float, z_med: float, parallax: float,
                     scene_scale_mode: str, scene_scale_stride: int,
                     device: str = "cuda", dtype=torch.float32, preprocess: bool = True,
                     depth_outliers: str = "gaussian", ignore_sky_mask: bool = False,
                     allow_empty_dynamic_mask: bool = False):
    """이미 로드된 recon 에서 cloud 를 굽되 **디스크를 안 거친다**. -> `load_cloud` 와 같은 dict.

    왜 이게 있나 (2026-09-11 사용자 지시 "cloud 로 만들되 저장은 하지 않는거지"). `cloud.npz` 는
    영상당 **1.4 GiB** 다 (dynpose 447편에 610.6 GiB). 디스크를 거치는 유일한 이유는 tau
    (`sample_camera_bank.py`) 와 fit (`fit_hole_ladder.py`) 이 **별개 프로세스**라 메모리를 못
    넘기기 때문이지, 재구축이 비싸서가 아니다. 실측 (00e9f728, 44.6 M points, GPU 0):

        load_recon_and_seg  4.74s   preprocess_recon  9.37s   build_cloud  0.60s
        save_cloud          7.56s   load_cloud        6.46s   (npz 1445.9 MiB)

    unproject 자체는 **0.60초**다. 게다가 tau/fit 은 둘 다 이 함수를 부르기 전에 `load_scene` 으로
    recon 을 **이미** 읽고 있고, `S`/`z_med`/`parallax` 는 `scene_graph.json` 의 `scale` 에 이미
    들어 있다. 그래서 in-memory 재구축의 실제 추가 비용은 preprocess+unproject ≈ 10s 뿐이고,
    그 대신 cloud 단계(굽기 28s + 프로세스 기동) 하나가 통째로 사라진다.

    `S`/`z_med`/`parallax` 를 **인자로 받는** 이유: 여기서 다시 재면 `scene_scale` 이 stride 1 에서
    6초 걸리고, 무엇보다 그 값이 graph 와 갈릴 수 있다. 게이지가 두 군데서 계산되면 안 된다
    (`lbm/cloud.py:49` 주석과 같은 이유). 호출자가 graph 의 `scale` 블록을 그대로 넘긴다.
    """
    num_frames, height, width, _ = recon["video"].shape
    K = vista4d["intrinsics_to_K"](recon["intrinsics"])
    dyn_frac = float(recon["dynamic_mask"].mean())
    if not allow_empty_dynamic_mask:
        assert_dynamic_mask_nonempty(dyn_frac, video, eval_data)

    scene = preprocess_recon(recon, vista4d, depth_outliers=depth_outliers,
                             ignore_sky_mask=ignore_sky_mask) if preprocess else recon
    colors, points_world, visible, indices, _ = build_cloud(scene, vista4d, device=device, dtype=dtype)

    frame0 = np.asarray(recon["cam_c2w"], dtype=np.float64)[0]
    return {
        "colors": colors, "points_world": points_world, "visible": visible, "indices": indices,
        "meta": {
            "video": video, "num_frames": num_frames, "height": height, "width": width,
            "fps": float(recon["fps"]), "S": float(S), "z_med_frame0": float(z_med),
            "parallax_ratio": float(parallax),
            "scene_scale_mode": scene_scale_mode, "scene_scale_stride": int(scene_scale_stride),
            "frame0_rot_deg": float(np.degrees(np.arccos(
                np.clip((np.trace(frame0[:3, :3]) - 1) / 2, -1, 1)))),
            "frame0_offset": float(np.linalg.norm(frame0[:3, 3])),
            "preprocess": bool(preprocess), "depth_outliers": depth_outliers,
            "ignore_sky_mask": bool(ignore_sky_mask),
            "dynamic_mask_frac": dyn_frac,
            "num_dynamic": int((visible.sum(dim=1) == 1).sum()),
            "cam_c2w": np.asarray(recon["cam_c2w"]).astype(np.float32), "K": K.astype(np.float32),
        },
    }


def save_cloud(output_path: str, colors, points_world, visible, indices, meta: dict):
    """visible (n, f) bool 은 그대로 저장하면 n*f 바이트라 packbits 로 8배 줄인다."""
    makedirs(path.dirname(output_path), exist_ok=True)
    visible_np = visible.cpu().numpy()
    np.savez(
        output_path,
        format=CLOUD_FORMAT,
        colors=colors.cpu().numpy().astype(np.uint8),
        points_world=points_world.cpu().numpy().astype(np.float32),
        visible_packed=np.packbits(visible_np, axis=1),
        visible_num_frames=np.int32(visible_np.shape[1]),
        indices=indices.cpu().numpy().astype(np.int32),
        **{f"meta_{k}": v for k, v in meta.items()},
    )


def load_cloud(input_path: str, device: str = "cuda", dtype=torch.float32):
    data = np.load(input_path, allow_pickle=False)
    assert str(data["format"]) == CLOUD_FORMAT, f"알 수 없는 cloud format: {data['format']}"
    num_frames = int(data["visible_num_frames"])
    visible = np.unpackbits(data["visible_packed"], axis=1, count=num_frames).astype(bool)
    warm_linalg(device)  # 디스크 경로도 같은 예열이 필요하다 (build_cloud 와 동일 이유)
    return {
        "colors": torch.as_tensor(data["colors"], device=device).to(dtype),
        "points_world": torch.as_tensor(data["points_world"], device=device).to(dtype),
        "visible": torch.as_tensor(visible, device=device),
        "indices": torch.as_tensor(data["indices"], device=device),
        "meta": {k[5:]: data[k] for k in data.files if k.startswith("meta_")},
    }


def main(args):
    vista4d = import_vista4d(args.vista4d_root)

    recon_folder = path.join(args.eval_data, "eval_data", "recon_and_seg", args.video)
    output_path = path.join(args.output_root or path.join(CINEMATRAJ_ROOT, "out"), args.video, "cloud.npz")
    if path.isfile(output_path) and args.skip_done:
        print(f"skip (done): {output_path}")
        return

    recon = vista4d["load_recon_and_seg"](recon_folder)
    num_frames, height, width, _ = recon["video"].shape
    K = vista4d["intrinsics_to_K"](recon["intrinsics"])

    # 좌표 규약 검사 — 여기서 안 걸리면 하류 게이트가 전부 조용히 틀린다.
    # frame0 은 I 가 **아니다**: 저자 배포본 recon_and_seg 는 우리 `recon_da3.py:43` 의 frame0
    # 앵커를 거치지 않았다 (camel 은 frame0 이 회전 6.0deg / 이동 0.0086 만큼 떠 있다). world 는
    # "cameras.npz 가 말하는 그 좌표계"로 정의하고, frame0 오프셋은 meta 에 기록만 한다.
    # emit 의 rel[0]=I 는 `inv(P[0]) @ P` 라 world 원점이 어디든 성립하므로 문제되지 않는다.
    # (`frame0_rot_deg`/`frame0_offset` 은 `cloud_from_recon` 이 meta 에 넣는다.)
    assert abs(K[0, 0, 2] / width - 0.5) < 0.05 and abs(K[0, 1, 2] / height - 0.5) < 0.05,\
        f"principal point 가 중앙이 아니다: cx={K[0, 0, 2]} cy={K[0, 1, 2]} (W={width} H={height})"
    finite = np.isfinite(recon["depths"][0]) & (recon["depths"][0] > 0)
    non_sky = ~recon["sky_mask"][0]
    assert (finite & non_sky).sum() / max(non_sky.sum(), 1) > 0.95, "non-sky depth 유효율 < 0.95"

    # S / z_med 는 **전처리 전 raw depth** 로 잰다. preprocess 가 sky depth 를 SKY_DEPTH(1e3) 로
    # 덮어쓰기 때문에 순서를 바꾸면 게이지가 통째로 망가진다 (non-sky 로 걸러도 습관적으로 위험).
    S = scene_scale(recon["depths"], K, recon["sky_mask"], cam_c2w=recon["cam_c2w"],
                    mode=args.scene_scale_mode, stride=args.scene_scale_stride)
    z_med = float(np.median(recon["depths"][0][finite & non_sky]))
    plx = parallax_ratio(recon["cam_c2w"], z_med)

    # 굽기 본체는 `cloud_from_recon` 하나다 (§empty_dynmask 가드 포함). tau/fit 의 in-memory
    # 경로와 **같은 함수**를 써야 두 경로가 갈리지 않는다 — meta 한 필드만 달라도 `CloudRenderer`
    # 가 다른 게이지로 조용히 돈다.
    cloud = cloud_from_recon(
        recon, vista4d, video=args.video, eval_data=args.eval_data,
        S=S, z_med=z_med, parallax=plx,
        scene_scale_mode=args.scene_scale_mode, scene_scale_stride=args.scene_scale_stride,
        device=args.device, preprocess=args.preprocess, depth_outliers=args.depth_outliers,
        ignore_sky_mask=args.ignore_sky_mask,
        allow_empty_dynamic_mask=args.allow_empty_dynamic_mask)
    points_world, num_dynamic = cloud["points_world"], cloud["meta"]["num_dynamic"]
    frame0_rot_deg = cloud["meta"]["frame0_rot_deg"]

    save_cloud(output_path, cloud["colors"], points_world, cloud["visible"], cloud["indices"],
               meta=cloud["meta"])

    print(f"\n{'video':<18}{'points':>12}{'dynamic':>12}{'S':>10}{'z_med':>9}{'parallax':>10}{'f0_rot':>9}")
    print(f"{args.video:<18}{points_world.shape[0]:>12,}{num_dynamic:>12,}"
          f"{S:>10.4f}{z_med:>9.3f}{plx:>10.4f}{frame0_rot_deg:>9.2f}")
    print(f"-> {output_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)  # None = <CinemaTraj>/out
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    # 전처리 없이 unproject 하면 static 경계 누수가 49프레임 쌓여 동적 물체가 빗살로 뭉개진다.
    # `--no_preprocess` 는 그 raw 경로를 재현해보기 위한 것이지 실사용 옵션이 아니다.
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no_preprocess", dest="preprocess", action="store_false")
    parser.add_argument("--depth_outliers", default="gaussian", choices=["gaussian", "pool"])
    parser.add_argument("--ignore_sky_mask", action="store_true", default=False)

    # 빈 dynamic_mask 로 굽는 것은 **기본적으로 사고**다 (§empty_dynmask). 동적 물체가 정말
    # 없는 씬을 굽고 싶을 때만 켠다 — 켜면 meta 의 `dynamic_mask_frac` 이 0 으로 남아 나중에
    # 그 씬들만 골라낼 수 있다.
    parser.add_argument("--allow_empty_dynamic_mask", action="store_true", default=False)

    # S 의 정의. `build_scene_graph.py` 와 **같은 값이어야 한다** — 두 산출물(`scene_graph.json`,
    # `cloud.npz`)이 서로 다른 게이지를 들고 있으면 게이트 마진 `0.02·S` 와 후보 거리가 어긋난다.
    parser.add_argument("--scene_scale_mode", default="points_first_cam", choices=SCALE_MODES)
    parser.add_argument("--scene_scale_stride", default=1, type=int)   # 픽셀 서브샘플 (1 = 전부)

    main(parser.parse_args())
