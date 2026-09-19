"""GenDoP 릴리즈 ckpt 를 **우리 코퍼스의 val split** 에 돌려 우리 모델 arm 들과 한 표에 올린다.

왜 스크립트인가: 단계가 4개(inputs → infer → eval_dir → score)이고 그 사이에 경로가 6개
흐른다. 손으로 부르면 어느 arm 이 어느 캡션·어느 depth 게이지로 돌았는지가 셸 히스토리에만
남는다. 여기서는 `CORPORA` 표가 곧 config 이고 `--stage` 로 중간부터 다시 돌 수 있다.

이 파일은 `tmp/d156/run_gendop_d121.py`(vista 전용)를 코퍼스 인자화해서 `scripts/` 로 승격한
것이다. **vista_d121 항목은 그때 실제로 돈 값 그대로** 이므로 같은 명령이 같은 결과를 낸다.
새 코퍼스는 `CORPORA` 에 항목 하나를 더하는 것으로 끝난다.

기존 스크립트는 한 줄도 안 고친다. 네 개를 그대로 조합한다:
  scripts/dynpose_gendop_inputs.py     rgbd/<scene>/<idx>/rgbd/{frame_0000.png,frame_depth_0000.npy}
      + text_motion/ 캡션 미러(열거용 — 실제 텍스트는 --text_from_eval_dir 이 덮는다)
  scripts/gendop_release_infer.py      --text_from_eval_dir 로 우리 eval 폴더의
      `<prefix>_<scene>_<idx>_caption.json` 을 읽는다 = latentcam 이 받은 문장 그대로
  scripts/gendop_preds_to_eval_dir.py  native pose → 49 index-pick, rmax 를 GT 에서 빌림
  scripts/eval_subject_in_frame.py     --eval_dir LABEL=DIR 를 arm 마다 하나씩

`--pose_length` — 모델이 **직접** 뽑을 포즈 수. 30 은 릴리즈 학습 길이(기본, 예전 런과 동일),
49 는 우리 코퍼스 길이다. 49 로 주면 `--pose_length 49 --no_strict_pose_length` 가 나가고
출력이 `pred_<tag>_p49` 로 갈라져 30 짜리 런을 안 덮는다. `stage_evaldir` 의 `--src_poses` 도
같이 따라가므로 리샘플(→49 index-pick)이 no-op 이 된다.

`depth_norm` 은 코퍼스마다 정한다 — **frame0 depth 의 median 이 GenDoP 학습 대역(MonST3R
~0.32; 릴리즈 `text_rgbd/case1_depth.npy` 는 0.266~0.722 median 0.323)에 있는가**가 기준이다.
  vista d121   DA3 depth,           per-scene median 2.5~4.4                → `median`
  dynpose d137 recon_and_seg EXR,   per-scene median 0.41~20.55 (med 2.68)  → `median`
`dynpose_gendop_inputs.py` 요약표의 **`min` 행(0.19~6.16)을 median 으로 읽으면 안 된다** —
그 행은 프레임 최솟값의 통계다. 실제 median 행은 2.68 이라 대역 밖이고, `none` 으로 두면
깊이만 8배 큰 조건이 들어간다.

`--text_dir` / `--text_tag` — 텍스트 조건만 갈아끼운다. 기본은 ref arm 의 eval 폴더(=우리
모델이 실제로 받은 D121 자연어 문장)이고, `scripts/gendop_style_captions.py` 가 만든
GenDoP 분포 정합 문장(motion 태그 + target 만)을 넣으면 **문장 탓 / 모델 탓**이 갈린다.
`--text_tag` 를 안 주면 기본 캡션 산출물을 덮어쓰므로 둘은 같이 준다.

env: inputs/infer 는 `GenDoP` (cv2 EXR), eval_dir 는 아무거나, score 는 `vista4d`.

사용:
    cd models/Planner/CinemaTraj
    python scripts/run_gendop_eval.py --corpus dynpose_d137 --stage inputs
    python scripts/run_gendop_eval.py --corpus dynpose_d137 --stage infer \
        --arm gendop_rgbd --pose_length 49 --gpu 0 --limit 4      # smoke
    python scripts/run_gendop_eval.py --corpus dynpose_d137 --stage infer \
        --arm gendop_rgbd --pose_length 49 --gpu 0
    python scripts/run_gendop_eval.py --corpus dynpose_d137 --stage evaldir \
        --arm gendop_rgbd --pose_length 49
    python scripts/run_gendop_eval.py --corpus dynpose_d137 --stage score \
        --arm gendop_rgbd --pose_length 49 --gpu 0
"""
from argparse import ArgumentParser
from os import environ, makedirs, path
from subprocess import run

HERE = path.dirname(path.dirname(path.abspath(__file__)))       # .../CinemaTraj
EVAL_MY = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/eval_my/"
CKPT = "/data1/cympyc1785/pipeline/GenDoP/checkpoints"
TEXT_KEY = "Concise Interaction"      # 우리 eval 폴더 caption json 의 유일한 키

# `eval_data` 규약이 두 스크립트에서 다르다. 상수 하나를 양쪽에 넘기면 경로가 겹쳐 죽는다.
#   dynpose_gendop_inputs.py:74   `<eval_data>/recon_and_seg/<scene>`      → eval_data 까지
#   eval_subject_in_frame.py      → scene_graph/io.py:88 이 다시 "eval_data" 를 붙인다 → 그 위
# 그래서 `eval_data` 하나만 적고 score 쪽은 그 부모를 쓴다.
CORPORA = {
    # d156 에서 실제로 돈 설정 그대로 (2026-09-06, 875 entry)
    "vista_d121": dict(
        corpus="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121",
        split_name="seg_list_vista4d_test.txt",
        eval_data="/data1/cympyc1785/data/Vista4D-Eval-Data/eval_data",
        prefix="vista4d",
        cloud_root=path.join(HERE, "out"),
        out=path.join(HERE, "results", "20260906_d156_gendop_d121"),
        depth_norm="median",       # DA3 depth median 2.5~4.4 → 대역 밖
        # 라벨은 config 의 실제 조작축을 따른다 (D124 는 molmo2 단독이 아니다):
        #   D123 vista4d_d121_da3_t128     geo_encoder=da3   video_latent_dim=0     → da3 only
        #   D133 vista4d_d121_molmo2_nogeo geo_encoder=null  video_latent_dim=2560  → molmo2 only
        #   D124 vista4d_d121_molmo2       geo_encoder=da3   video_latent_dim=2560  → da3+molmo2
        ours={
            "d123_da3":        EVAL_MY + "20260904_165917_vista4d_d121_da3_t128__epoch100__seed42",
            "d133_molmo2":     EVAL_MY + "d133_d121_molmo2_nogeo__last",
            "d124_da3_molmo2": EVAL_MY + "20260904_185554_vista4d_d121_molmo2__epoch100__seed42",
        },
        ref="d123_da3",            # ref/caption 원본을 빌려올 arm
    ),
    # D162 (2026-09-07, 27 entry). vista_d121 과 **같은 코퍼스·같은 ckpt** 인데 split 만 다르다 —
    # snowboard 는 학습 split 이라 `seg_list_vista4d_test.txt` 에 없다. `eval_testset.py
    # --set test_seg_list=` 로 track_* 27 entry 만 따로 뽑은 eval 폴더를 `ours` 로 쓴다.
    # 빠르게 움직이는 subject 에서 tracking 이 되는지 보려는 것이라 preset 은 track_* 만 골랐다.
    "vista_d121_snowboard": dict(
        corpus="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121",
        split_name="seg_list_vista4d_snowboard_track27.txt",
        eval_data="/data1/cympyc1785/data/Vista4D-Eval-Data/eval_data",
        prefix="vista4d",
        cloud_root=path.join(HERE, "out"),
        out=path.join(HERE, "results", "20260907_d162_gendop_snowboard"),
        depth_norm="median",
        ours={
            "d123_da3":        EVAL_MY + "d162_snowboard27__20260904_165917_vista4d_d121_da3_t128__epoch100",
            "d124_da3_molmo2": EVAL_MY + "d162_snowboard27__20260904_185554_vista4d_d121_molmo2__epoch100",
        },
        ref="d123_da3",
    ),
    # D158 (2026-09-07, 1129 entry). 세 arm 전부 epoch 100 last.pth.
    "dynpose_d137": dict(
        corpus="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d137",
        split_name="seg_list_dynpose_dd10_test.txt",
        eval_data="/data1/cympyc1785/data/DynPose-LBM/eval_data",
        prefix="dynpose",
        cloud_root=path.join(HERE, "out_dynpose"),
        out=path.join(HERE, "results", "20260907_d158_gendop_d137"),
        depth_norm="median",       # EXR depth per-scene median 0.41~20.55 (med 2.68) → 대역 밖
        ours={
            "d137_da3":        EVAL_MY + "20260906_013323_dynpose_d137_da3__last",
            "d137_molmo2":     EVAL_MY + "20260906_024038_dynpose_d137_molmo2_nogeo__last",
            "d137_da3_molmo2": EVAL_MY + "20260906_030245_dynpose_d137_da3_molmo2__last",
        },
        ref="d137_da3",
    ),
    # D208 (2026-09-20, 5,144 entry = d200 test split 전량). 5-arm 학습이 쓰는 그 split 이다.
    #
    # **recon 루트가 다른 dynpose 세대와 다르다.** d200 은 d185+d199 pooled = dynpose-100k
    # 이라 recon 이 `DynPose-100K/eval_data` 에 있다. d137 까지 쓰던 `DynPose-LBM/eval_data`
    # 를 그대로 두면 1,017 씬 중 **107 씬**(두 코퍼스가 겹치는 몫)만 잡힌다 — 처음에 그 107 을
    # 보고 "rgbd 가 10% 뿐"이라고 잘못 읽었다. 실제로는 video.mp4/depths 둘 다 1,017/1,017 이라
    # `gendop_rgbd` 도 돌릴 수 있다. score 단계도 이 루트로 recon 을 읽는다.
    #
    # cloud.npz 는 0/1,017 이다 (D178 에서 디스크 저장을 없앴다) -> score 는
    # `--cloud_source memory` 여야 한다. npz 모드면 전 씬이 "cloud/graph 없음" 으로 빠져
    # 표가 통째로 nan 이 된다 (조용히 실패한다).
    "dynpose_d200": dict(
        corpus="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200",
        split_name="seg_list_dynpose_s91_test.txt",
        eval_data="/data1/cympyc1785/data/DynPose-100K/eval_data",
        prefix="dynpose",
        cloud_root=path.join(HERE, "out_dynpose"),
        out=path.join(HERE, "results", "20260920_d208_gendop_d200"),
        depth_norm="median",
        # D200 5-arm 중 epoch 49 last 로 testset eval 이 끝난 넷 (2026-09-18 run 이름 기준).
        # ⑤ molmo2_dec_l21 은 아직 학습 중이라 빠져 있다.
        ours={
            "d200_da3":             EVAL_MY + "20260918_140904_dynpose_d200_da3__last",
            "d200_molmo2":          EVAL_MY + "20260918_140909_dynpose_d200_molmo2_da3__last",
            "d200_molmo2_l21":      EVAL_MY + "20260918_140914_dynpose_d200_molmo2_l21_da3__last",
            "d200_molmo2_dec":      EVAL_MY + "20260918_140919_dynpose_d200_molmo2_dec_da3__last",
        },
        ref="d200_da3",
    ),
}

# GenDoP arm 2종. text 는 어느 ckpt 를 쓰든 --text_from_eval_dir 로 같은 문장을 받는다.
ARMS = [
    dict(tag="gendop_text", ckpt="text_motion.safetensors", cond="text"),
    dict(tag="gendop_rgbd", ckpt="text_rgbd.safetensors", cond="depth+image+text"),
]

PY_GENDOP = "/data1/cympyc1785/miniconda3/envs/GenDoP/bin/python"
PY_LATENT = "/data1/cympyc1785/miniconda3/envs/latentcam/bin/python"
PY_VISTA = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"


def sh(cmd, gpu=None, tag=""):
    env = dict(environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[{tag}] {' '.join(cmd[:3])} ...", flush=True)
    r = run(cmd, cwd=HERE, env=env)
    print(f"[{tag}] rc={r.returncode}", flush=True)
    return r.returncode


def split_path(C):
    return path.join(C["corpus"], C["split_name"])


def suffix_of(pose_length, text_tag=None):
    """30(릴리즈 기본)·기본 캡션은 예전 경로 그대로, 그 외는 접미사로 갈라 덮어쓰기를 막는다."""
    return ("" if pose_length == 30 else f"_p{pose_length}") + (f"_{text_tag}" if text_tag else "")


def pred_dir(C, arm, pose_length, text_tag=None):
    return path.join(C["out"], f"pred_{arm['tag']}{suffix_of(pose_length, text_tag)}")


def eval_suffix(pose_length, text_tag=None, rescale=True, resample="index_pick",
                scale_token=True):
    """eval 폴더·점수 접미사. infer 산출물(`pred_*`)은 rescale 과 무관하므로 여기서만 갈린다.

    `rescale=False`(=`--raw`)면 GenDoP 이 낸 **자기 크기 그대로** eval 폴더로 간다. 기본값
    True 는 `rmax` 를 GT 것으로 바꿔 심으므로 최대 변위가 GT 와 같아지고, 그러면 충돌률·
    subject_in_frame 처럼 **절대 크기에 반응하는 지표**가 GenDoP 이 아니라 GT 를 재게 된다
    (d121 rgbd p49 기준 median x2.48 확대였다).

    `resample` 도 eval 폴더에서만 갈린다 — 같은 npz 를 두 방식으로 읽는 것이라 접미사
    `_slerp` 로 나눠 index_pick 판본을 안 덮는다.

    `scale_token=False`(=`--no_scale_token`)가 **공식 배포 경로**다 — `eval.py:239
    pose_normalize` 는 scale 토큰이 안 걸린 pose 로 pred JSON 을 쓴다. 우리 infer 는 곱한
    `c2w` 와 곱한 값 `scale` 을 npz 에 같이 넣으므로 되나누면 정확히 배포 판본이 된다
    (median 0.50 배, 분포 폭 13.6배라 크기 지표에 그대로 나타난다).
    """
    return (suffix_of(pose_length, text_tag) + ("" if rescale else "_raw")
            + ("" if resample == "index_pick" else "_slerp")
            + ("" if scale_token else "_noscale"))


def stage_inputs(C, overwrite=False, text_only=False):
    makedirs(C["out"], exist_ok=True)
    cmd = [PY_GENDOP, "scripts/dynpose_gendop_inputs.py",
           "--split", split_path(C), "--corpus", C["corpus"],
           "--eval_data", C["eval_data"], "--out", C["out"]]
    if overwrite:
        cmd += ["--overwrite"]
    # 코퍼스가 `text_only=True` 면 rgbd 를 안 만든다 (dynpose_d200 처럼 recon 이 일부만 남은 경우).
    if text_only or C.get("text_only"):
        cmd += ["--text_only"]
    return sh(cmd, tag="inputs")


def stage_infer(C, gpu, limit=None, depth_norm=None, pose_length=30, arms=None,
                overwrite=False, text_dir=None, text_tag=None):
    depth_norm = depth_norm or C["depth_norm"]
    # 기본은 ref arm 의 eval 폴더(= latentcam 이 실제로 받은 문장). `--text_dir` 를 주면
    # 같은 모양의 다른 폴더로 갈아끼운다 (예: `gendop_style_captions.py` 가 만든 분포 정합 문장).
    text_dir = text_dir or C["ours"][C["ref"]]
    for arm in arms or ARMS:
        cmd = [PY_GENDOP, "scripts/gendop_release_infer.py",
               "--resume", path.join(CKPT, arm["ckpt"]),
               "--cond_mode", arm["cond"],
               "--text_key", TEXT_KEY,
               "--root", path.join(C["out"], "text_motion"),
               "--split", split_path(C),
               "--text_from_eval_dir", text_dir,
               # eval 폴더 파일명 접두사. 틀리면 전 엔트리가 조용히 skipped 로 빠진다
               "--eval_dir_prefix", C["prefix"],
               "--out", pred_dir(C, arm, pose_length, text_tag)]
        if pose_length != 30:
            # 학습 길이보다 길게 요구하는 것이므로 strict 폴백은 끈다 (짧게 끝나면 나온 만큼 디코드)
            # --forbid_eos 는 infer 쪽 auto (pose_length != 30 이면 on) 에 맡긴다
            cmd += ["--pose_length", str(pose_length), "--no_strict_pose_length"]
        if overwrite:
            cmd += ["--overwrite"]
        if arm["cond"] != "text":
            cmd += ["--gen_root", path.join(C["out"], "rgbd"), "--monst3r_sub", "rgbd",
                    "--rgbd_fit", "letterbox", "--depth_norm", depth_norm]
        if limit:      # smoke: split 앞 N 개 entry 만
            with open(split_path(C), encoding="utf-8") as f:
                only = [l.strip().split("/", 1)[1] for l in f if l.strip()][:limit]
            cmd += ["--only", *only]
        if sh(cmd, gpu=gpu, tag=arm["tag"]):
            return 1
    return 0


def stage_evaldir(C, pose_length=30, arms=None, text_tag=None, rescale=True,
                  resample="index_pick", scale_token=True, text_dir=None):
    suffix = eval_suffix(pose_length, text_tag, rescale, resample, scale_token)
    # gendop_slerp 은 GenDoP 의 core/utils 를 import 하므로 그 env 로 돌려야 한다
    py = PY_LATENT if resample == "index_pick" else PY_GENDOP
    for arm in arms or ARMS:
        cmd = [py, "scripts/gendop_preds_to_eval_dir.py",
               "--pred_dir", pred_dir(C, arm, pose_length, text_tag),
               "--ref_eval_dir", C["ours"][C["ref"]],
               "--out_dir", path.join(C["out"], f"eval_dir_{arm['tag']}{suffix}"),
               "--prefix", C["prefix"], "--npz_kind", "latentcam",
               # pose_length=49 면 src==n_poses 라 리샘플이 항등이 된다
               "--src_poses", str(pose_length), "--n_poses", "49",
               "--resample", resample,
               "--caption_from", "ref"]
        # 텍스트를 갈아끼운 런은 캡션도 그 폴더에서 가져와야 caption f-score 가 **모델이 받은**
        # 문장을 읽는다. 안 주면 ref eval 폴더 = 예전 런과 비트 동일.
        if text_dir:
            cmd += ["--caption_dir", text_dir]
        if not rescale:
            cmd += ["--no_rescale"]
        if not scale_token:
            cmd += ["--no_scale_token"]
        if sh(cmd, tag=arm["tag"]):
            return 1
    return 0


def stage_score(C, gpu, pose_length=30, arms=None, text_tag=None, rescale=True,
                resample="index_pick", scale_token=True, extra_eval_dir=(),
                scenes=None, subject_occlusion=False, out_tag=None,
                cloud_source="npz"):
    """`arms` 가 빈 리스트면 GenDoP arm 없이 `--extra_eval_dir` 만 올린다 (E.T. 같은 외부 베이스라인).

    `arms=None` 이 "전부"(기존 동작)이고 `arms=[]` 가 "하나도 없음"이라 `or ARMS` 로는 못 가른다.
    """
    suffix = eval_suffix(pose_length, text_tag, rescale, resample, scale_token)
    cmd = [PY_VISTA, "scripts/eval_subject_in_frame.py",
           "--ref_eval_dir", C["ours"][C["ref"]], "--name_prefix", C["prefix"],
           "--cloud_root", C["cloud_root"],
           "--corpus_root", C["corpus"], "--eval_data", path.dirname(C["eval_data"]),
           # D178 이후 dynpose 는 cloud.npz 를 디스크에 안 남긴다 — npz 모드면 전 씬이
           # "cloud/graph 없음" 으로 조용히 빠져 표가 통째로 nan 이 된다.
           "--cloud_source", cloud_source]
    for label, d in C["ours"].items():
        cmd += ["--eval_dir", f"{label}={d}"]
    for arm in (ARMS if arms is None else arms):
        cmd += ["--eval_dir",
                f"{arm['tag']}{suffix}="
                + path.join(C["out"], f"eval_dir_{arm['tag']}{suffix}")]
    # 우리 파이프라인 밖에서 만든 eval 폴더(E.T./DIRECTOR 등)를 같은 표에 올린다.
    for spec in extra_eval_dir:
        cmd += ["--eval_dir", spec]
    if scenes:
        # 씬 표본을 파일로 고정한다 — D205 표와 같은 200 씬을 써야 열끼리 비교가 된다.
        cmd += ["--scenes", scenes]
    if subject_occlusion:
        cmd += ["--subject_occlusion"]
    cmd += ["--out", path.join(C["out"],
                               f"subject_in_frame{out_tag or suffix}.json")]
    return sh(cmd, gpu=gpu, tag="score")


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="vista_d121", choices=sorted(CORPORA))
    ap.add_argument("--stage", default="all",
                    choices=["inputs", "infer", "evaldir", "score", "all"])
    ap.add_argument("--gpu", default="0")               # GPU 0~4 만 (프로젝트 규칙)
    ap.add_argument("--limit", type=int, default=None)  # smoke 용 엔트리 수
    # 안 주면 코퍼스 기본값 (vista=median, dynpose=none) — depth median 이 대역 안이냐로 정한다
    ap.add_argument("--depth_norm", default=None, choices=["none", "median"])
    # 모델이 직접 뽑을 포즈 수. 30 = 릴리즈 학습 길이(기본, 예전 런과 동일), 49 = 우리 코퍼스 길이
    ap.add_argument("--pose_length", type=int, default=30)
    # 한 arm 만 돌릴 때. 안 주면 ARMS 전부
    # `none` 은 GenDoP arm 을 하나도 안 올린다 — score 단계에서 `--extra_eval_dir` 만 볼 때.
    ap.add_argument("--arm", default=None, choices=[a["tag"] for a in ARMS] + ["none"])
    # 이어달리기(기본)는 이미 있는 npz 를 건너뛴다. GenDoP 는 `generate_mode='sample'` 이고
    # seed 를 루프 **시작에 한 번** 심으므로, 중간부터 이으면 앞뒤 엔트리가 서로 다른 RNG
    # 스트림에서 나온다 — arm 전체가 config 로 재현이 안 된다. 중단 후 재개는 --overwrite.
    ap.add_argument("--overwrite", action="store_true")
    # inputs 단계에서 rgbd 를 안 만든다 (코퍼스가 `text_only=True` 면 자동으로 켜진다).
    ap.add_argument("--text_only", action="store_true")
    # 텍스트 조건을 갈아끼운다. 안 주면 ref arm 의 eval 폴더 = 예전 런과 비트동일.
    # `<dir>/test/<prefix>_<scene>_<idx>_caption.json` 모양이면 무엇이든 된다.
    ap.add_argument("--text_dir", default=None)
    # 산출물 경로 접미사 (`pred_gendop_rgbd_p49_gdstyle`). --text_dir 를 줄 때 같이 줄 것 —
    # 안 주면 기본 캡션으로 돈 결과를 덮어쓴다.
    ap.add_argument("--text_tag", default=None)
    # GenDoP 이 낸 크기를 그대로 쓸지(`--raw`), `rmax` 를 GT 것으로 바꿔 심을지(기본).
    # 기본값은 예전 런과 비트동일하게 두되, 절대 크기에 반응하는 지표는 `--raw` 로 잰다
    # (2026-09-07 사용자 지시 "앞으로 raw output 그대로 써주고").
    ap.add_argument("--rescale", dest="rescale", action="store_true", default=True)
    ap.add_argument("--raw", dest="rescale", action="store_false")
    # native pose 수를 49 로 늘리는 방법. index_pick = 예전 런과 비트동일(19 pose 가 두 번
    # 뽑혀 계단), gendop_slerp = GenDoP 원본 `core/utils.sample_from_dense_cameras`
    # (회전 SLERP + 이동 LERP). pose_length=49 면 둘 다 항등이라 차이가 없다.
    ap.add_argument("--resample", choices=["index_pick", "gendop_slerp"],
                    default="index_pick")
    # scale 토큰(`coords[:,9]`) 적용 여부. `--no_scale_token` 이 **공식 배포 판본**이다 —
    # 배포 `eval.py` 는 scale 을 궤적 PNG 에만 쓰고 pred JSON 에는 안 넣는다. 기본값 True 는
    # 예전 런과 비트동일용 (2026-09-07 사용자 지시로 지표는 `--no_scale_token` 으로 잰다).
    ap.add_argument("--scale_token", dest="scale_token", action="store_true", default=True)
    ap.add_argument("--no_scale_token", dest="scale_token", action="store_false")
    # score 단계 전용. 우리 파이프라인 밖 베이스라인(E.T./DIRECTOR)의 eval 폴더를 `LABEL=DIR` 로
    # 같은 표에 올린다. 여러 번 줄 수 있다.
    ap.add_argument("--extra_eval_dir", action="append", default=[])
    # 씬 표본 파일. D205 는 `tmp/d205/sample200_scenes.txt` 200 씬으로 쟀다 — 그 표와 열을
    # 나란히 놓으려면 같은 파일을 줘야 한다. 안 주면 test split 전 씬(=기존 동작).
    ap.add_argument("--scenes", default=None)
    # 가림 열(`subject_visible_frac`/`_min`). D205 표에는 켜져 있다.
    ap.add_argument("--subject_occlusion", action="store_true")
    # 점수 JSON 파일명 접미사. 안 주면 eval_suffix (기존 동작). 같은 suffix 로 표본만 바꿔
    # 여러 번 잴 때 덮어쓰기를 막는다.
    ap.add_argument("--out_tag", default=None)
    # dynpose 는 D178 이후 cloud.npz 가 디스크에 없다 -> `memory`. vista 는 아직 npz 가 있다.
    ap.add_argument("--cloud_source", default="npz", choices=["npz", "memory"])
    a = ap.parse_args()
    assert not (a.text_dir and not a.text_tag), "--text_dir 를 주면 --text_tag 도 줄 것"
    # infer/evaldir 는 `arms or ARMS` 라 빈 리스트가 조용히 "전부"로 되살아난다. score 전용.
    assert not (a.arm == "none" and a.stage != "score"), "--arm none 은 --stage score 에서만"

    C = CORPORA[a.corpus]
    arms = [] if a.arm == "none" else [x for x in ARMS
                                       if a.arm is None or x["tag"] == a.arm]
    rc = 0
    if a.stage in ("inputs", "all"):
        rc = rc or stage_inputs(C, a.overwrite, a.text_only)
    if a.stage in ("infer", "all"):
        rc = rc or stage_infer(C, a.gpu, a.limit, a.depth_norm, a.pose_length, arms,
                               a.overwrite, a.text_dir, a.text_tag)
    if a.stage in ("evaldir", "all"):
        rc = rc or stage_evaldir(C, a.pose_length, arms, a.text_tag, a.rescale, a.resample,
                                 a.scale_token, a.text_dir)
    if a.stage in ("score", "all"):
        rc = rc or stage_score(C, a.gpu, a.pose_length, arms, a.text_tag, a.rescale,
                               a.resample, a.scale_token, a.extra_eval_dir, a.scenes,
                               a.subject_occlusion, a.out_tag, a.cloud_source)
    print(f"\n{'corpus':<12}{a.corpus}\n{'out':<12}{C['out']}\n"
          f"{'arms':<12}{[x['tag'] for x in arms]}\n{'rc':<12}{rc}")


if __name__ == "__main__":
    main()
