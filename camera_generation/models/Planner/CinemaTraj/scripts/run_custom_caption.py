"""영상 파일 1개 + 캡션 1줄 → 우리 모델이 만든 카메라 + depth warp 영상.

사용자 지시 2026-09-21:
  "내가 영상 위치랑 caption text 직접 넣으면 우리 모델 돌려서 카메라 저장하고
   depth warp 영상도 만들어서 저장해주는 pipeline 간단하게 이어서 만들어서 내가 쓸 수 있게"

═══ 기존 경로와 무엇이 다른가 ═══════════════════════════════════════════════════════
`run_d215.py` 계열은 **뱅크(pseudo-GT 카메라)를 먼저 굽고** 그 카메라에서 캡션을 생성한다.
여기서는 캡션이 사람 손으로 들어오므로 뱅크·route·tau·fit·emit·instance_desc 가 전부
필요 없다. 남는 것은 "모델이 조건으로 읽는 것"뿐이다:

  ① 소스 프레임 49장 + 소스 카메라   ← recon (DA3)
  ② avg_scale S                      ← scene_graph 의 `scale` 블록
  ③ 캡션 텍스트                      ← 사용자가 준다
  ④ PE-AV(Molmo2) 캐시               ← ①③ 에서

`target_poses.npz` 는 **형식상 필요해서** 소스 궤적을 그대로 한 벌 넣는다 (dataset 이
(V,T,4,4) 를 요구한다). 이건 GT 가 아니므로 eval 이 찍는 clatr/caption 점수는
**읽지 말 것** — 이 파이프라인의 산출물은 예측 카메라와 warp 영상 두 개다.
`--stage score` 가 없는 이유가 이것이다.

═══ 단계 ═══════════════════════════════════════════════════════════════════════════
  recon     영상 → DA3 depth/pose + SAM3 dyn/sky 마스크 + seg_instances   (vista4d, GPU)
  geocalib  중력축 사이드카                                               (geocalib env, GPU)
  graph     scene_graph.json (여기서 쓰는 건 `scale.S` 와 `cameras.K`)    (vista4d)
  corpus    1-entry latentcam 코퍼스 (캡션이 여기 들어간다)               (이 파일)
  molmo2    PE-AV video/prefill 캐시                                      (latentcam, GPU)
  eval      seed 별 카메라 생성 (`last.pth`)                              (latentcam, GPU)
  warp      SOURCE | seed별 pred depth warp 릴                            (vista4d, GPU)
  bundle    번들 폴더로 정리 (원본 영상·캡션·카메라·warp 한 자리)
  all       위 전부

캡션이 두 군데로 들어간다 — 같은 문장이 아니다:
  · `prompts.json` 의 concise = 사용자 캡션 **원문** (umt5 text embedding 입력)
  · molmo2 override  = `Track {target}.`  (D200 학습 때의 override 생성기와 같은 형식)
`--target` 을 안 주면 캡션에서 지칭구를 뽑아 쓴다(§guess_target). 뽑은 결과를 항상
찍으므로 틀렸으면 `--target` 으로 고쳐서 다시 돌릴 것.

env: vista4d (recon/geocalib/graph/corpus/warp), latentcam (molmo2/eval)

예시:
  python scripts/run_custom_caption.py --stage all --gpu 1 \
      --video /data1/.../my_clip.mp4 --name my-clip \
      --caption "The camera tracks the grey car while orbiting to the right around it."

  # 지칭구를 직접 지정하고 seed 를 바꾸려면
  python scripts/run_custom_caption.py --stage all --gpu 1 \
      --video ... --name my-clip --caption "..." --target "the grey car" --seeds 42 7 99
"""
import re
import sys
from argparse import ArgumentParser
from csv import writer as csv_writer
from json import dump, load
from os import environ, listdir, makedirs, path, replace
from shutil import copyfile
from subprocess import run

import numpy as np

HERE = path.dirname(path.abspath(__file__))
CT = path.dirname(HERE)
ROOT = "/data1/cympyc1785/LatentCamVid"
LATENTCAM = path.join(ROOT, "camera_generation/latentcam")
VISTA4D = path.join(ROOT, "video_generation/models/Vista4D")

PY_VISTA = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"
PY_LATENTCAM = "/data1/cympyc1785/miniconda3/envs/latentcam/bin/python"
# GeoCalib 은 kornia 를 쓰는데 vista4d/latentcam/GenDoP/sam3 넷 다 없다 — 전용 env 가 따로 있다
# (`scripts/run_dynpose_d148_geocalib.sh` 도 같은 이유로 이 python 을 쓴다).
PY_GEOCALIB = "/data1/cympyc1785/miniconda3/envs/geocalib/bin/python"

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
HEIGHT, WIDTH = 720, 1280                # recon 해상도. 코퍼스는 --image_scale 로 반만 쓴다
IMAGE_SCALE = 0.5                        # 640x360 (vista4d_bank_to_dl3dv.py 와 같은 값)
CHUNK_PREFIX = "vista4d"                 # 하류(render_pred_depth_warp)의 `--name_prefix`


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


# ------------------------------------------------------------------------------ target
# 지칭구 추출. 우리 캡션 템플릿은 전부 "The camera <동사구> {T} <부가절>" 형태다
# (build_bank_captions.py). 아래 패턴은 그 템플릿에서 실제로 쓰이는 동사구만 적는다 —
# 못 맞히면 None 을 돌려주고 호출부가 캡션 원문으로 떨어뜨린다 (조용히 틀린 지칭구를
# 지어내는 것보다 낫다).
_TARGET_RE = re.compile(
    r"\bcamera\s+(?:slowly\s+|gently\s+|rapidly\s+|quickly\s+)?"
    r"(?:tracks|follows|orbits(?:\s+around)?|circles|pushes\s+in\s+(?:on|toward(?:s)?)|"
    r"pulls?\s+(?:back|out)\s+from|cranes?\s+(?:up(?:ward)?|down(?:ward)?)\s+(?:and\s+)?"
    r"(?:over|above)|drops?\s+straight\s+down\s+while\s+staying\s+on|"
    r"rises?\s+while\s+staying\s+on|stays?\s+on|keeps?\s+on|arcs?\s+around)\s+"
    r"(?P<target>.+?)"
    r"(?=\s*(?:,|\.|\bwhile\b|\bas\b|\brevealing\b|\bkeeping\b|\band\b|$))",
    re.IGNORECASE)


def guess_target(caption: str):
    """캡션 → 지칭구. 못 맞히면 None.

    왜 캡션 전체를 molmo2 override 로 안 쓰나: D200 학습 때 override 는 `Track {지칭구}.`
    한 형식뿐이었다 (`run_d215.py:stage_override`). 문장 전체를 넣으면 학습 분포 밖이라
    PE-AV 가 target 을 못 짚는다. 그래서 지칭구를 뽑아 같은 틀에 넣는다.
    """
    hit = _TARGET_RE.search(caption or "")
    if not hit:
        return None
    text = hit.group("target").strip().strip(",.")
    # "it" / "them" 은 지칭이 아니라 대명사다 — 그걸 Track 에 넣으면 아무 물체도 안 가리킨다.
    return None if text.lower() in ("it", "them", "him", "her", "the subject") else text


_ARTICLES = ("the ", "a ", "an ", "this ", "that ", "some ")


def guess_nouns(target):
    """지칭구 → SAM3 keyword 목록.

    왜 필요한가: recon 을 `--seg_keywords _all_` 로 돌리면 SAM3 자체를 안 타고
    (`recon_and_seg_single.py:82`) `--save_seg_instances` 가 조용히 무시된다. 그러면
    `scene_graph/io.py:117` 의 `assert segs` 에서 죽는다. vista 코퍼스도 전부
    metadata.csv 의 `dynamic` 열(명사 목록)로 SAM3 를 돌렸으므로 같은 규약이다.

    관사를 떼고, 형용사가 붙은 구면 머리명사도 같이 넣는다 ("the white dog" ->
    ["white dog", "dog"]) — 구가 너무 좁아 0 검출이 나는 쪽이 더 흔한 실패다.
    """
    text = (target or "").strip().lower().strip(",.")
    for art in _ARTICLES:
        if text.startswith(art):
            text = text[len(art):]
            break
    text = text.strip()
    if not text:
        return []
    head = text.split()[-1]
    return [text] if head == text else [text, head]


def nouns_of(args):
    return list(args.nouns) if args.nouns else guess_nouns(args.target or guess_target(args.caption))


# ------------------------------------------------------------------------------- recon
def stage_recon(args):
    """영상 → DA3 recon + SAM3 마스크 + seg_instances.

    `--seg_keywords` 에는 지칭구에서 뽑은 명사를 넣는다 (§guess_nouns). `_all_` 로 두면
    SAM3 를 안 타서 `--save_seg_instances` 가 무시되고, 그 결과 graph 단계가
    "seg_instances 가 없다" 로 죽는다. seg_instances 는 recon 폴더 안에 떨어지는데
    `scene_graph/io.py:103-110` 이 거기를 fallback 으로 보므로 symlink 는 필요 없다.
    """
    out = path.join(RECON_ROOT, args.name)
    if path.isfile(path.join(out, "cameras.npz")) and not args.force:
        print(f"[skip] recon 이미 있음: {out}")
        return
    assert path.isfile(args.video), f"영상이 없다: {args.video}"
    nouns = nouns_of(args)
    assert nouns, ("SAM3 keyword 를 못 정했다 — `--nouns dog person` 또는 `--target` 을 줄 것 "
                   f"(캡션에서 뽑은 지칭구: {args.target or guess_target(args.caption)!r})")
    print(f"[recon] seg_keywords = {nouns}")
    makedirs(out, exist_ok=True)
    sh([PY_VISTA, "-m", "scripts.preprocess.recon_and_seg_single",
        "--video_path", args.video, "--output_folder", out,
        "--seg_keywords", *nouns,
        "--recon_method", "da3",
        "--da3_model_id", "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "--da3_process_res", "896",
        "--height", HEIGHT, "--width", WIDTH, "--num_frames", NUM_FRAMES,
        "--save_seg_instances", "--save_vis"],
       env={"CUDA_VISIBLE_DEVICES": str(args.gpu)}, cwd=VISTA4D,
       log=path.join(tmp_dir(args.name), "recon.log"))


def stage_geocalib(args):
    """중력축 사이드카. 없으면 `build_scene_graph.py --gravity_source geocalib` 이 죽는다."""
    sidecar = path.join(CT, "out", args.name, "geocalib_gravity.json")
    if path.isfile(sidecar) and not args.force:
        print(f"[skip] geocalib 이미 있음: {sidecar}")
        return
    sh([PY_GEOCALIB, "scripts/geocalib_gravity.py", "--videos", args.name,
        "--eval_data", EVAL_DATA],
       env={"CUDA_VISIBLE_DEVICES": str(args.gpu)}, cwd=CT,
       log=path.join(tmp_dir(args.name), "geocalib.log"))


def stage_graph(args):
    """scene_graph.json. 여기서 실제로 쓰는 건 `scale.S` 와 `cameras.{K,cam_c2w_world}` 지만,
    warp 렌더러(`lbm/render.py:open_renderer --cloud_source memory`)도 이 파일의 `scale`
    블록을 읽으므로 반드시 있어야 한다."""
    graph = path.join(CT, "out", args.name, "scene_graph.json")
    if path.isfile(graph) and not args.force:
        print(f"[skip] scene_graph 이미 있음: {graph}")
        return
    sh([PY_VISTA, "scripts/build_scene_graph.py", "--video", args.name,
        "--eval_data", EVAL_DATA, "--no_skip_done"],
       cwd=CT, log=path.join(tmp_dir(args.name), "graph.log"))


# ------------------------------------------------------------------------------ corpus
def stage_corpus(args):
    """1-entry latentcam 코퍼스. 규약은 `vista4d_bank_to_dl3dv.py` 모듈 docstring 그대로다.

    뱅크가 없으므로 `target_poses.npz` 에는 **소스 궤적을 그대로** 한 벌 넣는다. 모델은
    target 을 조건으로 읽지 않으므로(생성 대상이다) 예측은 영향받지 않고, 배열 형태만
    맞추는 용도다. eval 이 그 위에서 계산하는 지표는 의미가 없다 (§모듈 docstring).
    """
    import imageio.v3 as iio                                              # noqa: PLC0415

    name = args.name
    root, scene_dir = corpus_root(name), path.join(corpus_root(name), CHUNK_PREFIX, name)
    da3, img = path.join(scene_dir, "da3"), path.join(scene_dir, "images_4")
    makedirs(da3, exist_ok=True)
    makedirs(img, exist_ok=True)

    with open(path.join(CT, "out", name, "scene_graph.json"), encoding="utf-8") as file:
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

    target = args.target or guess_target(args.caption)
    prompts = {"0": {
        "frame_idx": [0, n],
        "prompt_camera_with_scene_video": {"concise": args.caption},
        "variant_id": f"custom_{name}_0", "preset": args.preset, "preset_raw": args.preset,
        "aim": "look_at" if target else "free", "anchor_label": target or "",
        "caption_fields": {"target_text": target} if target else {},
        "source": "custom_caption",
    }}
    with open(path.join(da3, "prompts.json"), "w", encoding="utf-8") as file:
        dump(prompts, file, ensure_ascii=False, indent=1)

    # avg_scale 은 scene 상수이고 소스만으로 계산된다 → 누수 없음. 두 ref 디렉토리 모두에
    # 같은 값을 쓴다 (`dataset_cfg.AVG_SCALE_DIRS` 의 기본 조합).
    # `dataset_cfg` 는 latentcam 쪽에 있다 (`vista4d_bank_to_dl3dv.py` 와 같은 이유로
    # 표를 두 벌 두지 않고 원본을 직접 읽는다).
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

    # molmo2 override (`Track {지칭구}.`). 지칭구를 못 뽑았으면 캡션 원문으로 떨어진다.
    override = {f"{CHUNK_PREFIX}_{name}_0":
                f"Track {target}." if target else args.caption}
    makedirs(tmp_dir(name), exist_ok=True)
    with open(path.join(tmp_dir(name), "molmo2_text_track.json"), "w", encoding="utf-8") as file:
        dump(override, file, ensure_ascii=False, indent=1)

    print(f"{'corpus':14s} {root}")
    print(f"{'frames':14s} {n}  {w}x{h}")
    print(f"{'avg_scale S':14s} {avg_scale:.4f}")
    print(f"{'caption':14s} {args.caption!r}")
    print(f"{'target':14s} {target!r}"
          f"{'' if args.target else '   (캡션에서 추출 — 틀렸으면 --target 으로 지정)'}")
    print(f"{'molmo2 text':14s} {next(iter(override.values()))!r}")


# ------------------------------------------------------------------------- molmo2/eval
def stage_molmo2(args):
    """PE-AV 캐시. D200 학습 때와 같은 인자 조합이되 `--free_mode` 만 변이 수에 맞춘다.

    `--free_mode zero` 는 `aim=='free'` 인 변이가 **하나도 없으면** assert 로 죽는다
    (`cache_molmo2_embeddings.py:485`). 코퍼스가 10k 변이일 때는 항상 섞여 있어서 안 걸리는데
    여기는 변이가 1개라 지칭구를 뽑은 순간(aim=look_at) free 가 0개가 된다. free 변이가 0개면
    `zero` 와 `off` 의 산출물은 **같다** — zero 는 free 변이를 0 행으로 묶는 처리일 뿐이다.
    """
    name = args.name
    override = path.join(tmp_dir(name), "molmo2_text_track.json")
    assert path.isfile(override), f"{override} 없음 — --stage corpus 먼저"
    free_mode = "zero" if (args.target or guess_target(args.caption)) is None else "off"
    sh([PY_LATENTCAM, "cache_molmo2_embeddings.py", "--gpu", args.gpu,
        "--root", corpus_root(name), "--seg_prefix", seg_prefix(name),
        "--splits", "train,test",
        "--video_out", path.join(molmo_cache(name), "video"),
        "--text_out", path.join(molmo_cache(name), "text.pt"),
        "--prefill_out", path.join(molmo_cache(name), "prefill.pt"),
        "--text_override_json", override, "--extra_layer", "21",
        "--joint", "--free_mode", free_mode, "--free_aim", "free",
        "--decode_tokens", "900", "--decode_keep", "points",
        "--decode_points", "49", "--fps", "2.0"],
       cwd=path.join(LATENTCAM, "main"),
       env={"PYTHONPATH": f"{LATENTCAM}:.", "WANDB_MODE": "disabled",
            "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"},
       log=path.join(tmp_dir(name), "molmo2_cache.log"))


def stage_eval(args):
    """seed 하나당 카메라 한 벌. `--set` 으로 코퍼스만 갈아끼우고 나머지는 run 의
    `config.yaml` 을 그대로 물려받는다 (프로젝트 규약 §추론 1)."""
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
    """SOURCE | seed별 pred 한 줄짜리 depth warp 릴. GT 열은 없다 (GT 가 없으므로)."""
    name = args.name
    out_dir = path.join(tmp_dir(name), "reel")
    sh([PY_VISTA, "scripts/render_pred_depth_warp.py", "--video", name,
        *sum([["--eval_dir", f"s{s}={eval_dir(name, s)}"] for s in args.seeds], []),
        "--out_dir", out_dir,
        "--corpus_root", corpus_root(name), "--name_prefix", CHUNK_PREFIX,
        "--cloud_root", path.join(CT, "out"), "--eval_data", EVAL_DATA,
        "--cloud_source", "memory",
        "--entries", "0", "--with_source", "--fps", "12", "--no_reel"],
       env={"CUDA_VISIBLE_DEVICES": str(args.gpu)}, cwd=CT,
       log=path.join(tmp_dir(name), "warp.log"))
    print(f"[warp] {out_dir}  {sorted(listdir(out_dir))}")


def stage_bundle(args):
    """번들 폴더로 정리 — 사용자 지시 "카메라 생성 돌리는 것들 ... 여기 안에 다 생기게".

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
    graph = path.join(CT, "out", name, "scene_graph.json")
    if path.isfile(graph):
        copyfile(graph, path.join(scene_out, "scene_graph.json"))

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
        sh([PY_VISTA, "scripts/bank_to_vista4d_cams.py", "--video", name,
            "--tag_prefix", "", "--pred_kind", "pred", "--preds", *preds,
            "--cam_dir", cams, "--overwrite"],
           cwd=CT, log=path.join(tmp_dir(name), "cams.log"))

    warp = path.join(tmp_dir(name), "reel", f"{stem}__warp.mp4")
    if path.isfile(warp):
        copyfile(warp, path.join(slot, "warp.mp4"))

    with open(path.join(slot, "caption.json"), "w", encoding="utf-8") as file:
        dump({"caption": args.caption,
              "target": args.target or guess_target(args.caption),
              "preset": args.preset}, file, ensure_ascii=False, indent=1)
    with open(path.join(slot, "info.json"), "w", encoding="utf-8") as file:
        dump({"source": "run_custom_caption.py", "name": name,
              "video_in": path.abspath(args.video) if args.video else None,
              "run": RUN, "ckpt": CKPT, "seeds": list(args.seeds),
              "corpus_root": corpus_root(name), "entry": stem,
              "eval_dirs": {f"s{s}": eval_dir(name, s) for s in args.seeds}},
             file, ensure_ascii=False, indent=1)
    print(f"[bundle] {slot}")


STAGES = {"recon": stage_recon, "geocalib": stage_geocalib, "graph": stage_graph,
          "corpus": stage_corpus, "molmo2": stage_molmo2, "eval": stage_eval,
          "warp": stage_warp, "bundle": stage_bundle}
ALL_ORDER = ("recon", "geocalib", "graph", "corpus", "molmo2", "eval", "warp", "bundle")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[*sorted(STAGES), "all"])
    parser.add_argument("--name", required=True, type=str,
                        help="씬 이름 (폴더/entry 이름이 된다). `_` 금지 — 하류가 `_` 로 자른다")
    parser.add_argument("--video", default="", type=str, help="입력 영상 경로 (recon 단계에만 필요)")
    parser.add_argument("--caption", default="", type=str, help="조건으로 줄 문장 한 줄")
    # 지칭구를 직접 지정. 안 주면 캡션에서 뽑는다 (§guess_target). molmo2 override 에만 쓰인다.
    parser.add_argument("--target", default="", type=str)
    # SAM3 검출 명사. 안 주면 지칭구에서 뽑는다 (§guess_nouns). recon 단계에만 쓰인다.
    parser.add_argument("--nouns", nargs="*", default=None)
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
    # recon 도 캡션이 필요하다 — SAM3 keyword 를 지칭구에서 뽑기 때문 (§stage_recon).
    if any(s in stages for s in ("recon", "corpus", "bundle")):
        assert args.caption or args.target or args.nouns, "--caption 이 필요하다"
    makedirs(tmp_dir(args.name), exist_ok=True)
    for stage in stages:
        print(f"\n===== [{args.name}] {stage} =====", flush=True)
        STAGES[stage](args)


if __name__ == "__main__":
    main()
