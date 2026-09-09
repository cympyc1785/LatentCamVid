"""Vista4D latentcam 뱅크의 **분모(avg_scale)** 를 context 점군 전체 기준으로 다시 잰다.

왜 필요한가: 지금 `latentcam_da3_k6_d77` 이 쓰는 분모는 `scene_graph.json:scale.S` 이고, 그건
**frame0 한 장**의 non-sky 평균 ray 길이다. context range 는 소스 영상 전체 [0,49) 인데
분모만 첫 프레임에서 나왔다. 카메라가 움직여 새 영역이 들어오는 씬에서는 그 한 장이 구간 전체의
깊이를 대표하지 못한다. 여기서는 **context depth 전량을 unproject 한 점군**에서 첫 카메라까지의
거리 평균을 잰다 — 점 집합만 바꾸고 기준점(첫 카메라)은 그대로다.

정의는 `trumans_to_recon.py:avg_scale_first_cam` 을 **그대로 import 해서** 쓴다. 새로 짜면 sky
마스크 / ray 곱셈 / 픽셀 stride 중 하나가 조용히 어긋난다 (실제로 DL3DV 쪽 writer 들은 sky 를
안 빼고 conf>=P40 만 쓴다 — 같은 이름 아래 정의가 이미 셋이다).

    S      (기존) = mean over **frame0** non-sky px  of  z * ||K^-1 [u+.5, v+.5, 1]||
    ctxall (신규) = mean over **전 프레임** non-sky px of  ||p_world - c2w[0][:3,3]||

두 값 모두 DA3 단위이고 scene 상수라 한 영상의 전 변이가 같은 값을 쓴다. 소스 영상만으로
계산되므로 target 누수가 없고 추론 때 복원 가능하다는 성질도 같다.

출력은 `dataset_dl3dv.py` 가 읽는 자리 그대로:
    <out_root>/vista4d/<video>/da3/avg_scale_ctx_all_first_cam/<seg>.json   bare float
seg 키는 그 영상 `da3/target_poses.npz` 의 `keys` 를 그대로 쓴다 (변이 수만큼).

사용 예시:
  python scripts/make_avg_scale_vista4d_ctxall.py --dry_run
  python scripts/make_avg_scale_vista4d_ctxall.py --workers 8
  python scripts/make_avg_scale_vista4d_ctxall.py --videos camel avocado-slice --no_skip_done
"""
import json
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
from os import listdir, makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from scene_graph.io import import_vista4d                                    # noqa: E402
from scene_graph.scale import scene_scale                                    # noqa: E402
from scripts.trumans_to_recon import avg_scale_first_cam                     # noqa: E402

VISTA4D_ROOT_DEFAULT = path.normpath(path.join(
    CINEMATRAJ_ROOT, "..", "..", "..", "..", "video_generation", "models", "Vista4D"))
RECON_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/eval_data/recon_and_seg"
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"
OUT_ROOT_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d77"
# dataset_cfg.AVG_SCALE_DIRS['ctx_all_first_cam'] 과 같아야 한다. 여기서 다시 적는 이유는
# latentcam 쪽 import 를 끌어오면 hydra/torch 가 딸려와 이 스크립트가 무거워지기 때문.
REF_DIR_DEFAULT = "avg_scale_ctx_all_first_cam"


def worker(job):
    video, eval_data, vista4d_root, out_root, ref_dir, stride, skip_done = job
    try:
        da3 = path.join(out_root, "vista4d", video, "da3")
        keys = [str(k) for k in np.load(path.join(da3, "target_poses.npz"))["keys"]]
        out_dir = path.join(da3, ref_dir)
        if skip_done and path.isdir(out_dir) \
                and all(path.isfile(path.join(out_dir, f"{k}.json")) for k in keys):
            return {"video": video, "status": "skip", "n_var": len(keys)}

        vista4d = import_vista4d(vista4d_root)
        recon = vista4d["load_recon_and_seg"](path.join(eval_data, "eval_data",
                                                        "recon_and_seg", video))
        K = vista4d["intrinsics_to_K"](recon["intrinsics"]).astype(np.float64)
        cam_c2w = recon["cam_c2w"].astype(np.float64)

        # 기존 분모(frame0 S). 같은 로더에서 재계산해 배율을 같이 낸다 — 뱅크에 저장된 값과
        # 어긋나면 로더/규약이 달라진 것이므로 아래에서 assert 로 잡는다.
        # `mode="frame0_ray"` 를 **명시**한다. 저장값이 옛 게이지라 기본 모드로 재면 아래 assert 가
        # 전 영상에서 터진다. 여기 쓰임은 "옛 값 재현"이므로 정의가 바뀌어도 이쪽은 안 따라간다
        # (아래 `ratio` 열이 곧 옛→새 배율이다 — `avg_scale_first_cam` 이 새 정의와 같은 식이다).
        s_frame0 = scene_scale(recon["depths"], K, recon["sky_mask"], mode="frame0_ray")
        got = avg_scale_first_cam(recon["depths"], cam_c2w, K[0], recon["sky_mask"],
                                  dynamic_mask=None, stride=stride)
        value = float(got["avg_scale"])

        stored_path = path.join(da3, "avg_scale_context_first_cam", f"{keys[0]}.json")
        stored = float(json.load(open(stored_path, encoding="utf-8"))) \
            if path.isfile(stored_path) else float("nan")
        if np.isfinite(stored):
            assert abs(s_frame0 - stored) / stored < 1e-6, \
                f"{video}: 재계산 S {s_frame0} != 저장값 {stored} — 로더/규약이 달라졌다"

        makedirs(out_dir, exist_ok=True)
        for seg in keys:
            with open(path.join(out_dir, f"{seg}.json"), "w", encoding="utf-8") as file:
                json.dump(value, file)
        return {"video": video, "status": "ok", "n_var": len(keys), "n_frames": len(cam_c2w),
                "S": s_frame0, "ctxall": value, "ratio": value / s_frame0,
                "n_points": int(got["n_points"])}
    except Exception as error:                                              # noqa: BLE001
        return {"video": video, "status": "fail", "why": f"{type(error).__name__}: {error}"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)    # recon_and_seg 의 부모
    parser.add_argument("--recon_root", default=RECON_DEFAULT)       # 영상 목록만 여기서 읽는다
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--out_root", default=OUT_ROOT_DEFAULT)      # latentcam 레이아웃 루트
    parser.add_argument("--ref_dir", default=REF_DIR_DEFAULT)        # <scene>/da3/<여기>/
    parser.add_argument("--videos", nargs="*", default=["all"])
    # DL3DV `make_avg_scale_da3_*.py` / TRUMANS 와 같은 픽셀 stride. 바꾸면 값이 미세하게 달라져
    # arm 간 비교가 깨진다.
    parser.add_argument("--stride", default=2, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    scenes_root = path.join(args.out_root, "vista4d")
    if args.videos == ["all"]:
        videos = sorted(v for v in listdir(scenes_root)
                        if path.isfile(path.join(scenes_root, v, "da3", "target_poses.npz")))
    else:
        videos = list(args.videos)

    jobs, missing = [], []
    for video in videos:
        need = [path.join(scenes_root, video, "da3", "target_poses.npz"),
                path.join(args.eval_data, "eval_data", "recon_and_seg", video, "video.mp4")]
        gone = [path.basename(p) for p in need if not path.isfile(p)]
        if gone:
            missing.append((video, f"없음: {', '.join(gone)}"))
            continue
        jobs.append((video, args.eval_data, args.vista4d_root, args.out_root,
                     args.ref_dir, args.stride, args.skip_done))

    print(f"{'out_root':22s} {args.out_root}")
    print(f"{'ref_dir':22s} {args.ref_dir}")
    print(f"{'scenes':22s} {len(jobs)}")
    print(f"{'missing':22s} {len(missing)}")
    for video, why in missing:
        print(f"  - {video}: {why}")
    if args.dry_run:
        return

    with Pool(max(1, args.workers)) as pool:
        rows = pool.map(worker, jobs)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\n{'video':<26} {'n_var':>6} {'S(frame0)':>11} {'ctxall':>11} {'ratio':>7}")
    for row in sorted(rows, key=lambda r: r["video"]):
        if row["status"] == "ok":
            print(f"{row['video']:<26} {row['n_var']:6d} {row['S']:11.5f} "
                  f"{row['ctxall']:11.5f} {row['ratio']:7.4f}")
        else:
            print(f"{row['video']:<26} {row['status']:>6} {row.get('why', '')}")
    if ok:
        ratios = np.array([r["ratio"] for r in ok])
        print(f"\n{'ok / skip / fail':22s} {len(ok)} / "
              f"{sum(r['status'] == 'skip' for r in rows)} / "
              f"{sum(r['status'] == 'fail' for r in rows)}")
        print(f"{'ratio med/min/max':22s} {np.median(ratios):.4f} / "
              f"{ratios.min():.4f} / {ratios.max():.4f}")
        print(f"{'변이 총합':22s} {sum(r['n_var'] for r in ok)}")


if __name__ == "__main__":
    main()
