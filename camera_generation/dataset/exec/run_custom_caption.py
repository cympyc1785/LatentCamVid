"""영상 1개 + 캡션 1줄 → 우리 모델이 만든 카메라 + depth warp 영상. **추론 전용 최소 경로.**

사용자 지시 2026-09-21:
  "내가 영상 위치랑 caption text 직접 넣으면 우리 모델 돌려서 카메라 저장하고
   depth warp 영상도 만들어서 저장해주는 pipeline"
  "sam3, vlm 이런거 안돌리고 순수히 우리 카메라 생성 모델만 돌리는거야.
   Molmo2 input은 따로 쓰게 해줘"
  "따로 pipeline 만들어달라고 한건 최소한의 기능만 남긴 inference code였어"

═══ 무엇을 **안** 하는가 ═══════════════════════════════════════════════════════════
SAM3 · noun VLM · instance_desc VLM · geocalib · scene graph 노드 · 뱅크(route/tau/
fit/emit) — 전부 안 돈다. 저것들은 **pseudo-GT 카메라를 굽기 위한** 것이고, 여기서는
카메라를 모델이 만든다. 지표도 안 낸다 (GT 가 없으므로 `--stage score` 가 없다).

그래서 SAM3 를 안 탄 recon 이 나온다 → `dynamic_mask` 가 전부 0 이고 seg 인스턴스가
없다. 그 recon 을 그대로 통과시키려고 하류에 붙인 스위치가 셋이다:
  `load_scene(allow_no_seg=True)` / `--allow_no_seg` / `--allow_empty_dynamic_mask`.
셋 다 기본 off 라 기존 뱅크·릴 경로는 비트 동일하다.

그 점군은 전 점이 static (동적 점 0개) 이라 **시간 누적(TP) 렌더를 쓰면 안 된다** —
움직이는 물체가 49프레임 겹쳐 유령 다발이 된다 (my-clip 실측: dynamic 0/44,994,231,
골퍼가 스윙 전 구간 겹침). 그래서 warp 은 `--temporal_persistence auto` 가
`--allow_no_seg` 를 보고 NTP(그 프레임 점만)로 떨어진다. 대가는 시간 누적 포기 =
카메라가 소스에서 멀어질수록 구멍이 커지는 것이고, 겹침은 구조적으로 안 생긴다.

═══ 모델이 조건으로 읽는 것 (그래서 남은 단계가 이것뿐) ═════════════════════════════
  ① 소스 프레임 49장 + 소스 카메라   ← recon (DA3)
  ② avg_scale S                      ← depths 에서 직접 (scene_graph.scale.scene_scale)
  ③ 캡션 텍스트                      ← `--caption`      (umt5 text embedding)
  ④ PE-AV(Molmo2) 텍스트             ← `--molmo2_text`  (안 주면 캡션 원문)

`target_poses.npz` 는 **형식상 필요해서** 소스 궤적을 그대로 한 벌 넣는다 (dataset 이
(V,T,4,4) 를 요구한다). GT 가 아니다.

③ 과 ④ 는 서로 다른 인코더로 들어간다. D200 학습 때 Molmo2 override 는 `Track {지칭구}.`
한 형식뿐이었으므로 (`run_d215.py:stage_override`), PE-AV 가 target 을 짚게 하려면
`--molmo2_text "Track the grey car."` 처럼 **그 틀로** 주는 쪽이 분포 안이다. 캡션 원문을
넣어도 돌기는 한다.

═══ 단계 (`--stage`) ═══════════════════════════════════════════════════════════════
  recon     영상 → DA3 depth/pose (+ DA3 sky). SAM3 없음.        (vista4d, GPU)
  scale     `scene_graph.json` **최소본** — `scale`/`cameras` 블록만 (nodes 없음)
  corpus    1-entry latentcam 코퍼스 (캡션이 여기 들어간다)
  molmo2    PE-AV video/prefill 캐시                              (latentcam, GPU)
  eval      seed 별 카메라 생성 (`last.pth`)                      (latentcam, GPU)
  warp      SOURCE | seed별 pred depth warp 릴                    (vista4d, GPU)
  bundle    번들 폴더로 정리 (원본 영상·캡션·카메라·warp 한 자리)
  all       위 전부

env: vista4d (recon/scale/corpus/warp), latentcam (molmo2/eval)

예시:
  python exec/run_custom_caption.py --stage all --gpu 1 \
      --video /data1/.../my_clip.mp4 --name my-clip \
      --caption "The camera pedestals down while looking at a man playing golf." \
      --molmo2_text "Track the man playing golf."
"""
import sys
from argparse import ArgumentParser
from csv import writer as csv_writer
from json import dump, load
from os import environ, listdir, makedirs, path
from shutil import copyfile
from subprocess import run

import numpy as np

HERE = path.dirname(path.abspath(__file__))
CT = path.dirname(HERE)
if CT not in sys.path:
    sys.path.insert(0, CT)
ROOT = "/data1/cympyc1785/LatentCamVid"
LATENTCAM = path.join(ROOT, "camera_generation/latentcam")
VISTA4D = path.join(ROOT, "video_generation/models/Vista4D")

PY_VISTA = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"
PY_LATENTCAM = "/data1/cympyc1785/miniconda3/envs/latentcam/bin/python"

# recon 은 vista 코퍼스 규약 자리에 떨군다 (`scene_graph/io.py:88` 이 `eval_data/` 를 붙인다).
EVAL_DATA = "/data1/cympyc1785/data/Vista4D-Eval-Data"
RECON_ROOT = path.join(EVAL_DATA, "eval_data", "recon_and_seg")
# 코퍼스는 씬마다 따로 낸다 — 한 통에 모으면 seg_list 가 서로를 덮는다.
CORPUS_PARENT = path.join(EVAL_DATA, "latentcam_custom")
BUNDLES = path.join(CT, "results", "20260921_d221_bundles")
TMP_PARENT = path.join(ROOT, "tmp", "custom")

RUN = "20260918_140914_dynpose_d200_molmo2_l21_da3"
CKPT = "last.pth"                        # best.pth 금지 (프로젝트 규약)
SEEDS_DEFAULT = (42, 1234, 2026)
NUM_FRAMES = 49                          # 코퍼스 규약. 소스가 더 길면 가운데를 자른다.
HEIGHT, WIDTH = 720, 1280                # recon 해상도. 코퍼스는 IMAGE_SCALE 로 반만 쓴다
IMAGE_SCALE = 0.5                        # 640x360 (vista4d_bank_to_dl3dv.py 와 같은 값)
CHUNK_PREFIX = "vista4d"                 # 하류(render_pred_depth_warp)의 `--name_prefix`
SCALE_MODE, SCALE_STRIDE = "points_first_cam", 1   # S 의 정의 (build_scene_graph.py 기본값)
# 노드가 없는 그래프다. `load_graph` 가 이 format 을 모르면 거기서 죽는 게 맞다 —
# 뱅크 굽기에 이걸 먹이면 anchor 0 개로 조용히 빈 뱅크가 나온다.
GRAPH_FORMAT_MIN = "planner_scene_graph_minimal_v1"


def sh(cmd, env=None, cwd=None, log=None):
    """외부 명령 하나. `log` 를 주면 stdout/stderr 를 그리로 (전량 tee 금지)."""
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    full = dict(environ, **(env or {}))
    if log:
        makedirs(path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as file:
            proc = run([str(c) for c in cmd], env=full, cwd=cwd, stdout=file, stderr=file)
    else:
        proc = run([str(c) for c in cmd], env=full, cwd=cwd)
    assert proc.returncode == 0, f"rc={proc.returncode}  (로그: {log})"


# ---------------------------------------------------------------- 경로 (이름 하나로 파생)
def tmp_dir(name):
    return path.join(TMP_PARENT, name)


def corpus_root(name):
    return path.join(CORPUS_PARENT, name)


def seg_prefix(name):
    """`seg_list_<prefix>_{train,test}.txt` 의 가운데 토막. 씬 이름에 `-` 가 섞여도 무방하다."""
    return f"custom_{name}"


def meta_csv(name):
    return f"meta_custom_{name}.csv"


def graph_path(name):
    return path.join(CT, "out", name, "scene_graph.json")


def eval_dir(name, seed):
    """`eval_testset.py` 산출 위치. 이름에 씬과 seed 를 둘 다 물린다 — 안 물리면 eval 은
    "이미 있음"으로 건너뛰고 릴은 옛 예측을 그리는데 둘 다 rc=0 이라 안 들킨다
    (memory `eval-dir-name-must-carry-tag`)."""
    return path.join(LATENTCAM, "eval_my", f"custom_{name}_s{seed}__last")


def molmo_cache(name):
    return path.join(corpus_root(name), "molmo2_cache")


def entry_name(name):
    """코퍼스 entry 이름 (= eval 이 떨구는 파일 stem). 변이가 하나라 seg 는 항상 `0`."""
    return f"{CHUNK_PREFIX}_{name}_0"


def molmo2_text(args):
    """PE-AV 에 들어갈 문장. 안 주면 캡션 원문 (§모듈 docstring ④)."""
    return args.molmo2_text or args.caption


# ------------------------------------------------------------------------------- recon
def stage_recon(args):
    """영상 → DA3 depth/pose. **SAM3 는 안 돈다.**

    `--seg_keywords` 를 **값 없이** 준다 — 기본값이 `["_all_"]` 이라 빼먹으면 정반대로
    전 픽셀이 dynamic 이 된다. 빈 목록이면 `recon_and_seg_single.py:75` 분기로 떨어져
    `dynamic_mask` 가 전부 0 이 된다 (`static_mask` 는 `media.py:190` 이 `~dynamic_mask`
    로 채우므로 전부 True). 같은 분기가 `sky_mask` 도 0 으로 덮으므로
    `--keep_recon_sky` 로 DA3 가 낸 sky 를 살린다 — 안 살리면 하늘의 무한 depth 가
    점군에 섞여 S 와 warp 가 같이 망가진다. (`_all_` 은 반대로 **전 픽셀을 dynamic**
    으로 만드는 분기라 여기 쓰면 안 된다.)
    """
    out = path.join(RECON_ROOT, args.name)
    if path.isfile(path.join(out, "cameras.npz")) and not args.force:
        print(f"[skip] recon 이미 있음: {out}")
        return
    assert path.isfile(args.video), f"영상이 없다: {args.video}"
    makedirs(out, exist_ok=True)
    sh([PY_VISTA, "-m", "scripts.preprocess.recon_and_seg_single",
        "--video_path", args.video, "--output_folder", out,
        "--seg_keywords",                       # 값 없음 = SAM3 안 탐 (기본값 `_all_` 를 덮는다)
        "--recon_method", "da3",
        "--da3_model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "--da3_process_res", "896",
        "--height", HEIGHT, "--width", WIDTH, "--num_frames", NUM_FRAMES,
        "--keep_recon_sky"],
       env={"CUDA_VISIBLE_DEVICES": str(args.gpu)}, cwd=VISTA4D,
       log=path.join(tmp_dir(args.name), "recon.log"))


def stage_scale(args):
    """`scene_graph.json` 최소본 — `scale` 과 `cameras` 블록만.

    왜 파일로 내는가: 코퍼스(`stage_corpus`)와 warp 렌더러(`lbm/render.py:331`)가 둘 다
    이 경로에서 게이지를 읽는다. 여기서 한 번만 계산해 박아 두면 두 소비자가 같은 S 를
    본다 — 각자 재면 게이지가 조용히 갈린다.

    노드·중력축·관계는 없다. `format` 을 일부러 다르게 찍어서 `load_graph` 가 이걸
    진짜 그래프로 오인하지 못하게 한다 (§GRAPH_FORMAT_MIN).
    """
    from scene_graph.io import load_scene                                  # noqa: PLC0415
    from scene_graph.scale import parallax_ratio, scene_scale, z_median    # noqa: PLC0415

    out = graph_path(args.name)
    if path.isfile(out) and not args.force:
        print(f"[skip] scene_graph 이미 있음: {out}")
        return
    recon = load_scene(EVAL_DATA, args.name, VISTA4D, allow_no_seg=True)
    num_frames, height, width, _ = recon["video"].shape
    K, cam_c2w = recon["K"], recon["cam_c2w"]
    S = scene_scale(recon["depths"], K, recon["sky_mask"], cam_c2w=cam_c2w,
                    mode=SCALE_MODE, stride=SCALE_STRIDE)
    z_med = z_median(recon["depths"], recon["sky_mask"])
    graph = {
        "format": GRAPH_FORMAT_MIN, "video": args.name,
        "num_frames": num_frames, "height": height, "width": width,
        "fps": float(recon["fps"]),
        "scale": {"S": S, "z_med_frame0": z_med,
                  "parallax_ratio": parallax_ratio(cam_c2w, z_med),
                  "mode": SCALE_MODE, "stride": SCALE_STRIDE,
                  "unit": "1 u = S DA3 units (all-frame non-sky points, "
                          "mean distance from first source camera)"},
        "cameras": {"cam_c2w_world": cam_c2w.tolist(), "K": K.tolist()},
        "nodes": [], "edges": [],
        "source": "run_custom_caption.py (추론 전용 · SAM3 없음 · 노드 없음)",
    }
    makedirs(path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        dump(graph, file, ensure_ascii=False, indent=1)
    print(f"{'graph':14s} {out}")
    print(f"{'frames':14s} {num_frames}  {width}x{height} @ {graph['fps']:.3g} fps")
    print(f"{'S':14s} {S:.4f}   z_med {z_med:.3f}   plx {graph['scale']['parallax_ratio']:.4f}")


# ------------------------------------------------------------------------------ corpus
def stage_corpus(args):
    """1-entry latentcam 코퍼스. 규약은 `vista4d_bank_to_dl3dv.py` 모듈 docstring 그대로다.

    뱅크가 없으므로 `target_poses.npz` 에는 **소스 궤적을 그대로** 한 벌 넣는다. 모델은
    target 을 조건으로 읽지 않으므로(생성 대상이다) 예측은 영향받지 않고, 배열 형태만
    맞추는 용도다.
    """
    import imageio.v3 as iio                                              # noqa: PLC0415

    name = args.name
    root, scene_dir = corpus_root(name), path.join(corpus_root(name), CHUNK_PREFIX, name)
    da3, img = path.join(scene_dir, "da3"), path.join(scene_dir, "images_4")
    makedirs(da3, exist_ok=True)
    makedirs(img, exist_ok=True)

    with open(graph_path(name), encoding="utf-8") as file:
        graph = load(file)
    src_c2w = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)
    n = src_c2w.shape[0]

    # `--fixed_focal` 규약: frame0 K 를 전 프레임에 복사한 뒤 이미지 배율만큼 줄인다.
    # cx/cy 를 같이 안 줄이면 hw_list 가 조용히 어긋난다.
    K = np.asarray(graph["cameras"]["K"], dtype=np.float64)[0].copy()
    K[:2, :] *= IMAGE_SCALE
    h, w = int(round(K[1, 2] * 2)), int(round(K[0, 2] * 2))

    frames = iio.imread(path.join(RECON_ROOT, name, "video.mp4"), plugin="FFMPEG")
    assert len(frames) == n, f"video {len(frames)} != pose {n}"
    if IMAGE_SCALE != 1.0:
        from PIL import Image                                             # noqa: PLC0415
        frames = [np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) for f in frames]
    for i, frame in enumerate(frames):
        iio.imwrite(path.join(img, f"{i:05d}.png"), frame)

    np.savez(path.join(da3, "pose.npz"),
             extrinsics=np.linalg.inv(src_c2w)[:, :3, :4].astype(np.float32),
             intrinsics=np.repeat(K[None], n, axis=0).astype(np.float32))
    # (V=1, T, 4, 4) w2c. 자리 채우기 — GT 아님 (§stage_corpus docstring).
    np.savez(path.join(da3, "target_poses.npz"),
             extrinsics=np.linalg.inv(src_c2w)[None].astype(np.float32),
             intrinsics=np.repeat(K[None, None], 1, axis=0).repeat(n, axis=1).astype(np.float32),
             keys=np.array(["0"]), variant_id=np.array([f"custom_{name}_0"]))

    prompts = {"0": {
        "frame_idx": [0, n],
        "prompt_camera_with_scene_video": {"concise": args.caption},
        "variant_id": f"custom_{name}_0", "preset": args.preset, "preset_raw": args.preset,
        # 뱅크가 없어 anchor 노드도 없다. `aim` 은 molmo2 `--free_mode` 분기에만 쓰인다.
        "aim": "look_at", "anchor_label": "", "caption_fields": {},
        "source": "custom_caption",
    }}
    with open(path.join(da3, "prompts.json"), "w", encoding="utf-8") as file:
        dump(prompts, file, ensure_ascii=False, indent=1)

    # avg_scale 은 scene 상수이고 소스만으로 계산된다 → 누수 없음. 두 ref 디렉토리 모두에
    # 같은 값을 쓴다 (`dataset_cfg.AVG_SCALE_DIRS` 의 기본 조합).
    sys.path.insert(0, path.join(LATENTCAM, "main"))
    from dataset_cfg import AVG_SCALE_DIRS                                # noqa: PLC0415
    avg_scale = float(graph["scale"]["S"])
    for ref in ("context_first_cam", "centroid"):
        ref_dir = path.join(da3, AVG_SCALE_DIRS[ref])
        makedirs(ref_dir, exist_ok=True)
        with open(path.join(ref_dir, "0.json"), "w") as file:
            dump(avg_scale, file)

    chunk = f"{CHUNK_PREFIX}/{name}"
    with open(path.join(root, meta_csv(name)), "w", newline="") as file:
        out = csv_writer(file)
        out.writerow(["chunk", "height", "width", "num_images"])
        out.writerow([chunk, h, w, n])
    # train 을 비우면 dataset 이 죽으므로 같은 줄을 양쪽에 둔다. 학습이 아니라 eval 전용
    # 코퍼스이고 `eval_testset.py` 는 test 목록만 읽으므로 누수 개념이 없다.
    for side in ("train", "test"):
        with open(path.join(root, f"seg_list_{seg_prefix(name)}_{side}.txt"), "w") as file:
            file.write(f"{chunk}/0\n")

    override = {entry_name(name): molmo2_text(args)}
    makedirs(tmp_dir(name), exist_ok=True)
    with open(path.join(tmp_dir(name), "molmo2_text.json"), "w", encoding="utf-8") as file:
        dump(override, file, ensure_ascii=False, indent=1)

    print(f"{'corpus':14s} {root}")
    print(f"{'frames':14s} {n}  {w}x{h}")
    print(f"{'avg_scale S':14s} {avg_scale:.4f}")
    print(f"{'caption':14s} {args.caption!r}")
    print(f"{'molmo2 text':14s} {molmo2_text(args)!r}"
          f"{'' if args.molmo2_text else '   (--molmo2_text 미지정 → 캡션 원문)'}")


# ------------------------------------------------------------------------- molmo2/eval
def stage_molmo2(args):
    """PE-AV 캐시. D200 학습 때와 같은 인자 조합.

    `--free_mode off`: `zero` 는 `aim=='free'` 인 변이가 **하나도 없으면** assert 로 죽는다
    (`cache_molmo2_embeddings.py:485`). 여기는 변이가 1개이고 `aim` 을 `look_at` 으로
    고정했으므로 free 가 0개다. free 변이가 0개면 `zero` 와 `off` 의 산출물은 **같다**.
    """
    name = args.name
    override = path.join(tmp_dir(name), "molmo2_text.json")
    assert path.isfile(override), f"{override} 없음 — --stage corpus 먼저"
    sh([PY_LATENTCAM, "cache_molmo2_embeddings.py", "--gpu", args.gpu,
        "--root", corpus_root(name), "--seg_prefix", seg_prefix(name),
        "--splits", "train,test",
        "--video_out", path.join(molmo_cache(name), "video"),
        "--text_out", path.join(molmo_cache(name), "text.pt"),
        "--prefill_out", path.join(molmo_cache(name), "prefill.pt"),
        "--text_override_json", override, "--extra_layer", "21",
        "--joint", "--free_mode", "off", "--free_aim", "free",
        "--decode_tokens", "900", "--decode_keep", "points",
        "--decode_points", "49", "--fps", "2.0"],
       cwd=path.join(LATENTCAM, "main"),
       env={"PYTHONPATH": f"{LATENTCAM}:.", "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
       log=path.join(tmp_dir(name), "molmo2_cache.log"))


def stage_eval(args):
    """seed 하나당 카메라 한 벌. `--set` 으로 코퍼스만 갈아끼우고 나머지는 run 의
    `config.yaml` 을 그대로 물려받는다 (프로젝트 규약 §추론 1).

    eval 이 같이 찍는 clatr/caption 점수는 **읽지 말 것** — target 이 소스 궤적 자리채움이라
    GT 가 아니다 (§stage_corpus)."""
    name = args.name
    for seed in args.seeds:
        out = eval_dir(name, seed)
        if path.isdir(path.join(out, "test")) and not args.force:
            print(f"[skip] seed {seed} 이미 있음: {out}")
            continue
        root = corpus_root(name)
        sh([PY_LATENTCAM, "scripts/eval_testset.py",
            "--run", path.join("results", RUN), "--ckpt", CKPT,
            "--gpu", args.gpu, "--sample-seed", seed, "--out", out,
            "--set", f"dl3dv_root={root}",
            "--set", f"meta_csv={meta_csv(name)}",
            "--set", f"train_seg_list={path.join(root, f'seg_list_{seg_prefix(name)}_train.txt')}",
            "--set", f"test_seg_list={path.join(root, f'seg_list_{seg_prefix(name)}_test.txt')}",
            "--set", f"peav_video_cache_dir={path.join(molmo_cache(name), 'video')}",
            "--set", f"peav_text_cache={path.join(molmo_cache(name), 'prefill.pt')}",
            # `--set` 값은 YAML 로 파싱된다 — `None` 은 문자열이라 가드를 통과한다. `null` 이어야 꺼진다.
            "--set", "text_emb_cache_path=null", "--set", "geo_raw_cache_dir=null"],
           cwd=LATENTCAM, log=path.join(tmp_dir(name), f"eval_s{seed}.log"))


# -------------------------------------------------------------------------- warp/bundle
def stage_warp(args):
    """SOURCE | seed별 pred 한 줄짜리 depth warp 릴. GT 열은 없다 (GT 가 없으므로).

    `--allow_no_seg --allow_empty_dynamic_mask`: SAM3 를 안 돌렸으니 인스턴스도 동적
    마스크도 없다. 그 점군은 동적 점이 0개라 `--temporal_persistence auto` 가 NTP 로
    떨어진다 — 안 그러면 움직이는 물체가 49프레임 겹친다 (§모듈 docstring).
    """
    name = args.name
    out_dir = path.join(tmp_dir(name), "reel")
    sh([PY_VISTA, "viz/render_pred_depth_warp.py", "--video", name,
        *sum([["--eval_dir", f"s{s}={eval_dir(name, s)}"] for s in args.seeds], []),
        "--out_dir", out_dir,
        "--corpus_root", corpus_root(name), "--name_prefix", CHUNK_PREFIX,
        "--cloud_root", path.join(CT, "out"), "--eval_data", EVAL_DATA,
        "--cloud_source", "memory", "--allow_no_seg", "--allow_empty_dynamic_mask",
        "--entries", "0", "--with_source", "--fps", "12", "--no_reel"],
       env={"CUDA_VISIBLE_DEVICES": str(args.gpu)}, cwd=CT,
       log=path.join(tmp_dir(name), "warp.log"))
    print(f"[warp] {out_dir}  {sorted(listdir(out_dir))}")


def stage_bundle(args):
    """번들 폴더로 정리.

        <BUNDLES>/<name>/source.mp4
        <BUNDLES>/<name>/<preset>/caption.json
        <BUNDLES>/<name>/<preset>/info.json
        <BUNDLES>/<name>/<preset>/warp.mp4
        <BUNDLES>/<name>/<preset>/cameras/s<seed>_transforms.json   (+ .npz)
    """
    name = args.name
    scene_out = path.join(BUNDLES, name)
    slot = path.join(scene_out, args.preset)
    cams = path.join(slot, "cameras")
    makedirs(cams, exist_ok=True)

    src = path.join(RECON_ROOT, name, "video.mp4")
    if path.isfile(src):
        copyfile(src, path.join(scene_out, "source.mp4"))

    stem = entry_name(name)
    preds = []
    for seed in args.seeds:
        pred = path.join(eval_dir(name, seed), "test", f"{stem}_transforms_pred.json")
        if not path.isfile(pred):
            continue
        copyfile(pred, path.join(cams, f"s{seed}_transforms.json"))
        preds.append(f"{eval_dir(name, seed)}:{stem}=s{seed}")
    # transforms.json -> cameras.npz (Vista4D `render_eval` 이 읽는 형식). 카메라만 따로
    # 쓰고 싶을 때를 위해 같이 낸다. `--preds` 는 `<eval_dir>:<entry>[=<tag>]` 형식이고
    # entry 는 **파일 stem 전체**다 (`vista4d_<name>_0`), seg 번호가 아니다.
    if preds:
        sh([PY_VISTA, "fit/convert/bank_to_vista4d_cams.py", "--video", name,
            "--tag_prefix", "", "--pred_kind", "pred", "--preds", *preds,
            "--cam_dir", cams, "--overwrite"],
           cwd=CT, log=path.join(tmp_dir(name), "cams.log"))

    warp = path.join(tmp_dir(name), "reel", f"{stem}__warp.mp4")
    if path.isfile(warp):
        copyfile(warp, path.join(slot, "warp.mp4"))

    with open(path.join(slot, "caption.json"), "w", encoding="utf-8") as file:
        dump({"caption": args.caption, "molmo2_text": molmo2_text(args),
              "preset": args.preset}, file, ensure_ascii=False, indent=1)
    with open(path.join(slot, "info.json"), "w", encoding="utf-8") as file:
        dump({"source": "run_custom_caption.py", "name": name,
              "video_in": path.abspath(args.video) if args.video else None,
              "run": RUN, "ckpt": CKPT, "seeds": list(args.seeds),
              "corpus_root": corpus_root(name), "entry": stem,
              "eval_dirs": {f"s{s}": eval_dir(name, s) for s in args.seeds}},
             file, ensure_ascii=False, indent=1)
    print(f"[bundle] {slot}")


STAGES = {"recon": stage_recon, "scale": stage_scale, "corpus": stage_corpus,
          "molmo2": stage_molmo2, "eval": stage_eval,
          "warp": stage_warp, "bundle": stage_bundle}
ALL_ORDER = ("recon", "scale", "corpus", "molmo2", "eval", "warp", "bundle")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[*sorted(STAGES), "all"])
    parser.add_argument("--name", required=True, type=str,
                        help="씬 이름 (폴더/entry 이름이 된다). `_` 금지 — 하류가 `_` 로 자른다")
    parser.add_argument("--video", default="", type=str, help="입력 영상 경로 (recon 단계에만 필요)")
    parser.add_argument("--caption", default="", type=str,
                        help="umt5 text embedding 으로 들어갈 문장 한 줄")
    # PE-AV(Molmo2) 에 들어갈 문장. 캡션과 **다른 인코더**라 따로 받는다 (§모듈 docstring ④).
    parser.add_argument("--molmo2_text", default="", type=str,
                        help="PE-AV 입력 문장 (안 주면 --caption 원문). 예: 'Track the grey car.'")
    # 번들 하위 폴더 이름. 카메라 자체는 preset 과 무관하다 (캡션이 전부다) — 이름표일 뿐.
    parser.add_argument("--preset", default="custom", type=str)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--force", action="store_true", default=False,
                        help="이미 있는 산출물도 다시 만든다")
    args = parser.parse_args()

    assert "_" not in args.name, (
        f"--name 에 `_` 를 쓰지 말 것 ({args.name}): 하류가 `vista4d_<name>_<seg>` 를 "
        "마지막 `_` 에서 잘라 seg 를 읽는다")
    stages = ALL_ORDER if args.stage == "all" else (args.stage,)
    if any(s in stages for s in ("corpus", "bundle")):
        assert args.caption or args.molmo2_text, "--caption 이 필요하다"
    makedirs(tmp_dir(args.name), exist_ok=True)
    for stage in stages:
        print(f"\n===== [{args.name}] {stage} =====", flush=True)
        STAGES[stage](args)


if __name__ == "__main__":
    main()
