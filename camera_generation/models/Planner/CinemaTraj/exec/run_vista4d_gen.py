"""영상 + 카메라 -> Vista4D 생성 영상. 한 줄로.

`run_custom_caption.py` 가 "영상 + 캡션 -> 카메라" 라면 이쪽은 그 뒤를 잇는다:
**이미 있는 카메라**(우리 모델 예측이든 뱅크든 손으로 만든 것이든)를 Vista4D 에 먹여
영상을 뽑는다.

    python exec/run_vista4d_gen.py \
        --video /path/to/source.mp4 \
        --camera <...>/bundles/car-roundabout/track_orbit_right/cameras/s1234.npz \
        --prompt "A dark gray Mini Cooper navigates a roundabout." --gpu 1

씬이 이미 recon 되어 있으면 `--video <씬이름>` 만 줘도 된다 (recon 단계를 건너뛴다).

## 단계 (`--stage`, 기본 all)

    recon   영상 -> DA3 depth/mask/cameras.npz   (`eval_data/recon_and_seg/<name>/`)
            이미 있으면 건너뛴다. 새 영상이면 SAM3 keyword 가 필요하다 (`--nouns`) —
            dynamic_mask 가 비면 render_eval 이 동적 물체를 배경처럼 끌고 간다.
    cam     카메라 파일 -> `eval_data/cameras/<name>/<tag>.npz`
            (`bank_to_vista4d_cams.py --cams`, 규약 변환은 전부 거기 한 군데)
    gen     `results/20260819_vista4d_eval/run_eval_gen.sh` (공식 2단계: render -> inference)
    out     생성 mp4 + source|gen 가로 concat 릴을 `--out_dir` 로

## 카메라 파일 두 형식

  * `.npz`  — `cam_c2w` (n,4,4) OpenCV c2w, **recon world 절대 미터**, frame0 = 소스 카메라.
              번들 `cameras/*.npz` 가 이 형식이다.
  * `.json` — nerfstudio(OpenGL c2w). 우리 모델 평가 산출물 `*_transforms_pred.json` 이 이것.

둘 다 **프레임 수가 recon 과 같아야** 하고 frame0 위치가 소스와 같아야 한다 (허용 0.05 m).
어긋나면 world 가 다른 것이라 `bank_to_vista4d_cams.py` 가 거기서 죽는다.

env: `vista4d` (recon/cam/gen 전부). GPU 1장.
"""
import csv
import json
import subprocess
import sys
from argparse import ArgumentParser
from os import environ, makedirs, path

ROOT = "/data1/cympyc1785/LatentCamVid"
CT = path.join(ROOT, "camera_generation/models/Planner/CinemaTraj")
V4 = path.join(ROOT, "video_generation/models/Vista4D")
EVAL_GEN = path.join(ROOT, "video_generation/results/20260819_vista4d_eval")
EVAL_DATA = "/data1/cympyc1785/data/Vista4D-Eval-Data"
ED = path.join(EVAL_DATA, "eval_data")
PY_V4 = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"

# Wan diffusion seed. 카메라만 바꿔 비교할 때는 **같은 값**이어야 한다 — 노이즈까지 다르면
# 차이가 카메라에서 온 건지 못 가른다.
SEED_DEFAULT = "52106"
OUT_DEFAULT = path.join(CT, "results/20260921_vista4d_custom")


def sh(cmd, log=None, **kw):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    if log is None:
        rc = subprocess.call(list(map(str, cmd)), **kw)
    else:
        makedirs(path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(list(map(str, cmd)), stdout=fh,
                                 stderr=subprocess.STDOUT, **kw)
    assert rc == 0, f"rc={rc}" + (f"  로그: {log}" if log else "")


def tmp_dir(name):
    return path.join(ROOT, "tmp", "vista4d_gen", name)


def recon_dir(args):
    return path.join(ED, "recon_and_seg", args.name)


def stage_recon(args):
    """영상 -> DA3 recon. 이미 있으면 건너뛴다.

    `--seg_keywords` 는 실제 명사여야 한다 — `_all_` 로 두면 SAM3 자체를 안 타서
    (`recon_and_seg_single.py:82`) dynamic_mask 가 비고, 동적 물체가 배경 점군에 섞인다.
    """
    out = recon_dir(args)
    if path.isfile(path.join(out, "cameras.npz")) and not args.force:
        print(f"[skip] recon 이미 있음: {out}")
        return
    assert path.isfile(args.video), (
        f"recon 이 없는데 `--video` 가 파일이 아니다: {args.video}\n"
        f"  새 영상이면 mp4 경로를, 이미 굽힌 씬이면 그 이름을 줄 것")
    assert args.nouns, "새 영상이면 `--nouns dog person` 처럼 SAM3 keyword 를 줘야 한다"
    makedirs(out, exist_ok=True)
    sh([PY_V4, "-m", "scripts.preprocess.recon_and_seg_single",
        "--video_path", args.video, "--output_folder", out,
        "--seg_keywords", *args.nouns, "--recon_method", "da3",
        "--num_frames", args.num_frames, "--height", args.height, "--width", args.width,
        "--save_vis"],
       env={**environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": V4},
       cwd=V4, log=path.join(tmp_dir(args.name), "recon.log"))


def stage_cam(args):
    sh([PY_V4, path.join(CT, "fit/convert/bank_to_vista4d_cams.py"),
        "--video", args.name, "--tag_prefix", "", "--overwrite",
        "--cams", f"{args.camera}={args.tag}"])


def meta_row(args):
    return {"name": f"{args.name}/{args.tag}", "video": args.name, "camera": args.tag,
            "seed": args.seed, "prompt": args.prompt, "dynamic": ",".join(args.nouns or []),
            "do_sky_seg": "false", "source": "custom", "video_id": ""}


def stage_gen(args):
    csv_path = path.join(tmp_dir(args.name), f"meta_{args.tag}.csv")
    makedirs(path.dirname(csv_path), exist_ok=True)
    row = meta_row(args)
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    sh(["bash", path.join(EVAL_GEN, "run_eval_gen.sh")], cwd=EVAL_GEN,
       env={**environ, "GPU": str(args.gpu), "CSV": csv_path, "STAGE": "all",
            "RES": args.res},
       log=path.join(tmp_dir(args.name), f"gen_{args.tag}.log"))


def gen_mp4(args):
    return path.join(ED, "gen", args.name, args.tag, f"video_seed={args.seed}.mp4")


def stage_out(args):
    """생성 mp4 + `source | gen` 가로 concat 릴."""
    import imageio.v2 as imageio
    import numpy as np

    src = path.join(recon_dir(args), "video.mp4")
    gen = gen_mp4(args)
    assert path.isfile(gen), f"생성 결과가 없다: {gen}"
    out = path.join(args.out_dir, args.name, args.tag)
    makedirs(out, exist_ok=True)

    cols = [("source", src), ("vista4d", gen)]
    stacks = [(label, imageio.mimread(p, memtest=False)) for label, p in cols]
    n = min(len(f) for _, f in stacks)
    h = min(f[0].shape[0] for _, f in stacks)
    frames = []
    for i in range(n):
        tiles = []
        for _, fr in stacks:
            im = fr[i][..., :3]
            w = int(round(im.shape[1] * h / im.shape[0]))
            iy = (np.arange(h) * im.shape[0] / h).astype(int)
            ix = (np.arange(w) * im.shape[1] / w).astype(int)
            tiles.append(im[iy][:, ix])
        frames.append(np.concatenate(tiles, axis=1))
    # libx264 + yuv420p 는 홀수 폭에서 조용히 0바이트를 낸다.
    frames = [f[:f.shape[0] // 2 * 2, :f.shape[1] // 2 * 2] for f in frames]
    imageio.mimwrite(path.join(out, "reel.mp4"), frames, fps=args.fps,
                     codec="libx264", quality=8, macro_block_size=1)
    subprocess.check_call(["cp", gen, path.join(out, "video.mp4")])
    with open(path.join(out, "info.json"), "w", encoding="utf-8") as file:
        json.dump({"scene": args.name, "tag": args.tag, "camera": path.abspath(args.camera),
                   "prompt": args.prompt, "seed": args.seed, "res": args.res}, file,
                  indent=2, ensure_ascii=False)
    print(f"[out] {out}\n  video.mp4 / reel.mp4 (source | vista4d) / info.json")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True,
                        help="소스 mp4 경로, 또는 이미 recon 된 씬 이름")
    parser.add_argument("--camera", required=True, help="카메라 npz 또는 transforms.json")
    parser.add_argument("--prompt", default="", help="Wan 텍스트 프롬프트 (비우면 빈 문자열)")
    parser.add_argument("--name", default=None, help="씬 이름 (기본: --video 의 파일 stem)")
    parser.add_argument("--tag", default=None, help="카메라 이름 (기본: --camera 의 stem)")
    parser.add_argument("--nouns", nargs="*", default=None,
                        help="SAM3 keyword. 새 영상 recon 에 필요하고 metadata dynamic 열로도 간다")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", default=SEED_DEFAULT, help="Wan diffusion seed")
    parser.add_argument("--res", default="384p", choices=["384p", "720p"])
    parser.add_argument("--num_frames", default="49")
    parser.add_argument("--height", default="720")
    parser.add_argument("--width", default="1280")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out_dir", default=OUT_DEFAULT)
    parser.add_argument("--stage", nargs="+", default=["all"],
                        choices=["all", "recon", "cam", "gen", "out"])
    parser.add_argument("--force", action="store_true", default=False,
                        help="recon 이 있어도 다시 돈다")
    args = parser.parse_args()
    args.name = args.name or path.splitext(path.basename(args.video))[0]
    args.tag = args.tag or path.splitext(path.basename(args.camera))[0]

    stages = ["recon", "cam", "gen", "out"] if "all" in args.stage else args.stage
    for stage in stages:
        print(f"\n===== {stage}  [{args.name}/{args.tag}] =====", flush=True)
        {"recon": stage_recon, "cam": stage_cam,
         "gen": stage_gen, "out": stage_out}[stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
