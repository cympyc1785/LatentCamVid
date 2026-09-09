"""`eval_data/gen` 의 생성 영상들을 MonST3R 로 돌려 depth + 카메라를 추정한다.

WHY: 생성 결과가 "요청한 카메라대로 움직였는지"를 재구성으로 되짚으려면 GT 없이 영상만으로
     pose 를 뽑아야 한다. MonST3R 는 동적 장면을 전제로 설계돼 있어 DA3/VGGT 처럼 움직이는
     피사체에 pose 가 끌려가지 않는다. DataDoP 캡션이 쓰는 게이지와도 같은 계열이라
     (`caption_cameras_datadop.py` 참고) 추정 pose 를 그대로 캡션 파이프라인에 넣을 수 있다.

**MonST3R 리포는 0줄 수정한다.** `run_single.py` 는 `--input_dir` 를 주면 gradio 앱을 띄우지 않고
바로 재구성만 하지만 모듈 최상단에서 `import gradio` 를 한다 (gradio 는 `set_scenegraph_options`
와 `main_demo` 안에서만 쓰인다). 그래서 import 전에 `sys.modules['gradio']` 에 빈 모듈을 꽂고
`get_reconstructed_scene` 만 가져다 쓴다. `dust3r/cloud_opt/optimizer.py:125` 가 RAFT 가중치를
**상대경로**로 열기 때문에 MonST3R 루트로 `chdir` 한 뒤 호출한다.

필요한 가중치 2종 (둘 다 HF, 리포의 `download_ckpt.sh` 는 vendored 사본에 없다):
  - 모델: `Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt`  (hub id 로 바로 로드)
  - flow: `MemorySlices/Tartan-C-T-TSKH-spring540x960-M` 의 `model.safetensors` 를
          `third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth` 로 변환해 둘 것

출력은 영상이 있는 preset 폴더 밑 `monst3r/` 에 모인다:
  `pred_traj.txt` (TUM, c2w) · `pred_intrinsics.txt` (프레임별 K 9열) ·
  `frame_depth_%04d.npy` (metric 아닌 MonST3R 게이지) · `frame_%04d.png` (rgb) ·
  `dynamic_mask_%d.png` · `conf_%d.npy` · `scene.glb` · `run_meta.json`

사용 예시:
    conda run -n GenDoP python scripts/monst3r_gen_videos.py --device cuda:0
    conda run -n GenDoP python scripts/monst3r_gen_videos.py --num_shards 2 --shard_id 0
    conda run -n GenDoP python scripts/monst3r_gen_videos.py --scenes camel avocado-slice
"""
from argparse import ArgumentParser
from glob import glob
from os import path, makedirs, chdir, getcwd, rename
from types import ModuleType, SimpleNamespace
import json
import shutil
import sys
import time
import traceback

MONST3R_ROOT = ("/data1/cympyc1785/LatentCamVid/camera_generation/models/GenDoP"
                "/dataset/monst3r")
DEFAULT_ROOT = "/data1/cympyc1785/LatentCamVid/DATA/Vista4D-Eval-Data/eval_data"
MODEL_ID = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt"
FLOW_CKPT = "third_party/RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"
OUT_NAME = "monst3r"          # preset 폴더 밑에 만들 하위 폴더 이름
STAGE_NAME = "NULL"           # run_single 이 seq_name="NULL" 일 때 쓰는 폴더 이름


def import_monst3r():
    """gradio 를 stub 으로 막고 MonST3R 의 재구성 함수만 가져온다."""
    if "gradio" not in sys.modules:
        sys.modules["gradio"] = ModuleType("gradio")     # 앱 코드에서만 쓰이므로 껍데기면 충분
    if MONST3R_ROOT not in sys.path:
        sys.path.insert(0, MONST3R_ROOT)
    from run_single import get_reconstructed_scene                       # noqa: E402
    from dust3r.model import AsymmetricCroCo3DStereo                     # noqa: E402
    return get_reconstructed_scene, AsymmetricCroCo3DStereo


def collect_jobs(root, scenes, pattern):
    """gen/<scene>/<preset>/video_seed=*.mp4 를 모은다."""
    jobs = []
    for video in sorted(glob(path.join(root, "gen", "*", "*", pattern))):
        preset_dir = path.dirname(video)
        scene = path.basename(path.dirname(preset_dir))
        if scenes and scene not in scenes:
            continue
        jobs.append(dict(scene=scene, preset=path.basename(preset_dir),
                         video=video, preset_dir=preset_dir,
                         out_dir=path.join(preset_dir, OUT_NAME)))
    return jobs


def is_done(out_dir):
    """pose 와 depth 가 둘 다 있어야 완료로 본다 (중간에 죽은 폴더를 재사용하지 않기 위해)."""
    return (path.exists(path.join(out_dir, "pred_traj.txt"))
            and bool(glob(path.join(out_dir, "frame_depth_*.npy"))))


def run_one(job, model, recon_fun_factory, args):
    """한 영상을 재구성하고 stage 폴더를 `monst3r/` 로 옮긴다."""
    stage_dir = path.join(job["preset_dir"], STAGE_NAME)
    if path.exists(stage_dir):
        shutil.rmtree(stage_dir)                          # 앞선 실패 잔해

    inner = SimpleNamespace(weights=None, output_dir=job["preset_dir"])
    recon_fun = recon_fun_factory(inner)
    started = time.time()
    scene, _outfile, _imgs = recon_fun(
        filelist=[job["video"]],
        schedule=args.schedule,
        niter=args.niter,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=True,
        mask_sky=False,
        clean_depth=True,
        transparent_cams=False,
        cam_size=0.05,
        show_cam=True,
        scenegraph_type=args.scenegraph_type,
        winsize=args.winsize,
        refid=0,
        seq_name=STAGE_NAME,                              # save_folder = output_dir/NULL
        new_model_weights=None,                           # inner.weights 와 같아야 재로드 안 함
        temporal_smoothing_weight=args.temporal_smoothing_weight,
        translation_weight=args.translation_weight,
        shared_focal=args.shared_focal,
        flow_loss_weight=args.flow_loss_weight,
        flow_loss_start_iter=args.flow_loss_start_iter,
        flow_loss_threshold=args.flow_loss_threshold,
        use_gt_mask=False,
        fps=args.fps,
        num_frames=args.num_frames,
    )
    # run_single 이 저장하지 않는 두 가지를 추가로 남긴다 (동적 마스크 / confidence).
    scene.save_dynamic_masks(stage_dir)
    scene.save_conf_maps(stage_dir)
    elapsed = time.time() - started

    num_frames = len(scene.imgs)
    if path.exists(job["out_dir"]):
        shutil.rmtree(job["out_dir"])
    rename(stage_dir, job["out_dir"])

    meta = dict(video=path.relpath(job["video"], args.root), scene=job["scene"],
                preset=job["preset"], model=MODEL_ID, flow_ckpt=FLOW_CKPT,
                num_frames=num_frames, elapsed_sec=round(elapsed, 1),
                image_size=args.image_size, niter=args.niter,
                scenegraph_type=args.scenegraph_type, winsize=args.winsize,
                shared_focal=args.shared_focal,
                flow_loss_weight=args.flow_loss_weight,
                temporal_smoothing_weight=args.temporal_smoothing_weight,
                translation_weight=args.translation_weight,
                min_conf_thr=args.min_conf_thr,
                convention="pred_traj.txt = TUM c2w (tx ty tz qx qy qz qw), "
                           "MonST3R gauge (metric 아님)")
    with open(path.join(job["out_dir"], "run_meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    return num_frames, elapsed


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)                  # eval_data 루트
    parser.add_argument("--pattern", default="video_seed=*.mp4")         # 대상 영상 glob
    parser.add_argument("--scenes", nargs="*", default=None)             # 지정하면 그 씬만
    parser.add_argument("--device", default="cuda")                      # cuda:N
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224])
    parser.add_argument("--niter", type=int, default=300)                # global alignment 반복
    parser.add_argument("--schedule", default="linear")
    parser.add_argument("--min_conf_thr", type=float, default=1.1)
    parser.add_argument("--scenegraph_type", default="swinstride")
    parser.add_argument("--winsize", type=int, default=5)
    parser.add_argument("--temporal_smoothing_weight", type=float, default=0.01)
    parser.add_argument("--translation_weight", default="1.0")
    parser.add_argument("--flow_loss_weight", type=float, default=0.01)
    parser.add_argument("--flow_loss_start_iter", type=float, default=0.1)
    parser.add_argument("--flow_loss_threshold", type=float, default=25)
    parser.add_argument("--fps", type=int, default=0)                    # 0 = 원본 프레임 그대로
    parser.add_argument("--num_frames", type=int, default=200)           # 영상당 상한
    parser.add_argument("--shared_focal", action="store_true", default=True)
    parser.add_argument("--no_shared_focal", dest="shared_focal", action="store_false")
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)                  # >0 이면 앞에서 N개만
    args = parser.parse_args()

    jobs = collect_jobs(args.root, set(args.scenes or []), args.pattern)
    total = len(jobs)
    jobs = [job for i, job in enumerate(jobs) if i % args.num_shards == args.shard_id]
    if args.limit:
        jobs = jobs[: args.limit]

    get_reconstructed_scene, AsymmetricCroCo3DStereo = import_monst3r()
    import functools
    import torch

    cwd = getcwd()
    chdir(MONST3R_ROOT)                    # optimizer.py:125 가 RAFT 를 상대경로로 연다
    flow_path = path.join(MONST3R_ROOT, FLOW_CKPT)
    assert path.exists(flow_path), f"flow ckpt 없음: {flow_path}"

    model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_ID).to(args.device)
    model.eval()

    def recon_fun_factory(inner):
        return functools.partial(get_reconstructed_scene, inner, inner.output_dir,
                                 model, args.device, True, args.image_size)

    rows, skipped, failed = [], 0, []
    for index, job in enumerate(jobs):
        tag = f"{job['scene']}/{job['preset']}"
        if args.skip_done and is_done(job["out_dir"]):
            skipped += 1
            continue
        makedirs(job["preset_dir"], exist_ok=True)
        print(f"[{index + 1}/{len(jobs)}] {tag}", flush=True)
        try:
            num_frames, elapsed = run_one(job, model, recon_fun_factory, args)
            rows.append((tag, num_frames, elapsed))
        except Exception as error:                       # 한 편 실패로 전체를 버리지 않는다
            failed.append((tag, f"{type(error).__name__}: {error}"))
            print(f"FAIL {tag}: {type(error).__name__}: {error}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()

    chdir(cwd)
    print()
    print(f"{'entries (all shards)':<26}{total}")
    print(f"{'this shard':<26}{len(jobs)}  (shard {args.shard_id}/{args.num_shards})")
    print(f"{'reconstructed':<26}{len(rows)}")
    print(f"{'skipped (done)':<26}{skipped}")
    print(f"{'failed':<26}{len(failed)}")
    if rows:
        mean = sum(r[2] for r in rows) / len(rows)
        print(f"{'mean sec / video':<26}{mean:.1f}")
        print()
        print(f"{'scene/preset':<44}{'frames':>8}{'sec':>9}")
        for tag, num_frames, elapsed in rows:
            print(f"{tag:<44}{num_frames:>8}{elapsed:>9.1f}")
    for tag, reason in failed:
        print(f"FAILED {tag:<40}{reason}")


if __name__ == "__main__":
    main()
