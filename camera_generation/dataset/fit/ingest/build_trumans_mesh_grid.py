"""TRUMANS chunk → `mesh_grid.npz` (G1 mesh 경로의 입력). Blender 1회 + numpy 1회.

**왜 별도 스크립트인가.** G1 을 mesh 로 재려면 두 단계가 필요한데 서로 다른 파이썬에서 돈다:
`fit/ingest/trumans_export_mesh.py` 는 **Blender 내장 인터프리터**(`bpy`)에서, `lbm/mesh_collision.py`
는 **vista4d env**(scipy)에서 돈다. 그리고 둘 다 `.blend` 경로 · 프레임 범위 · 소스 카메라
npz 를 알아야 하는데, 그건 `tru_<rec8>_a<NN>_<tag>` 이름에서 recon 폴더를 되찾아야 나온다
(`bank_to_blender_poses.chunk_paths`). 그 해석을 러너 셸에 손으로 적으면 두 벌이 되고, 실제로
`min_sweep_deg` 가 두 곳에 적혔다가 15↔30 으로 어긋난 전례가 있다. 그래서 여기 한 벌만 둔다.

산출물은 `<output_root>/<video>/mesh_grid.npz` — `sample_camera_bank.py` /
`fit_hole_ladder.py` 의 `--collision_source mesh` 가 인자 없이 찾는 바로 그 기본 경로다
(`lbm.mesh_collision.resolve_mesh_grid`).

중간 산출물 `mesh_gt.npz` (삼각형 원본) 은 격자보다 훨씬 크다 (a00 실측 격자 13.7 MB). 격자만
있으면 게이트가 도므로 `--keep_tris` 없이는 지운다.

**비용** (a00, 502 objects, 49프레임): Blender export ~2분(로드 16 s + 프레임별 평가),
격자 88.6 s. chunk 당 한 번이고 이분법이 몇 번 돌든 다시 안 띄운다.

사용 예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY fit/ingest/build_trumans_mesh_grid.py --video tru_1d076f8c_a00_s3f0k6 \
        --output_root out_trumans
"""
import json
import subprocess
import sys
from argparse import ArgumentParser
from os import makedirs, path, remove

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

import numpy as np  # noqa: E402

from lbm.mesh_collision import build_grid  # noqa: E402
from fit.convert.bank_to_blender_poses import BLENDER, chunk_paths  # noqa: E402


def export_tris(blend: str, frames: tuple, out: str, blender: str, move_eps: float, log: str):
    """Blender 를 **한 번** 띄워 프레임별 evaluated 삼각형을 뽑는다."""
    command = [blender, "-b", blend, "--python",
               path.join(CINEMATRAJ_ROOT, "fit", "ingest", "trumans_export_mesh.py"), "--",
               "--frames", str(frames[0]), str(frames[1]), str(frames[2]),
               "--out", out, "--move_eps", str(move_eps)]
    with open(log, "w", encoding="utf-8") as file:
        rc = subprocess.call(command, stdout=file, stderr=subprocess.STDOUT)
    assert rc == 0 and path.isfile(out), f"mesh export 실패 (rc={rc}) — 로그: {log}"
    return out


def main(args):
    manifest_path, src_poses_path, blend = chunk_paths(args.video, args.recon_root)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    f0, f1 = manifest["frames"]
    step = int(manifest["frame_step"])
    folder = path.join(args.output_root, args.video)
    makedirs(folder, exist_ok=True)
    grid = args.out or path.join(folder, "mesh_grid.npz")
    tris = path.join(folder, "mesh_gt.npz")
    log = path.join(folder, "mesh_export.log")

    if path.isfile(grid) and args.skip_done:
        print(f"{'SKIP':<16}{grid} (이미 있음)")
        return
    if not (path.isfile(tris) and args.skip_done):
        print(f"{'blend':<16}{blend}")
        print(f"{'frames':<16}{f0}..{f1} step {step}")
        export_tris(blend, (f0, f1, step), tris, args.blender, args.move_eps, log)

    #    소스 카메라는 **blend world** 다. `build_grid` 가 이걸로 ① 격자 AABB 를 넓히고
    #    ② flood-fill 씨앗을 잡고 ③ `anchor_c2w = src[0]` 를 격자에 심는다 (뱅크 world ↔
    #    blend world 강체 앵커). 셋 다 같은 배열이라 인자를 하나로 뒀다.
    src = np.load(src_poses_path)["cam_c2w"].astype(np.float64)
    build_grid(tris, src, grid, voxel=args.voxel, pad=args.pad, clip=args.clip)
    if not args.keep_tris:
        remove(tris)
    print(f"\n{'grid':<16}{grid}  ({path.getsize(grid) / 1e6:.1f} MB)")
    print(f"{'tris':<16}{'kept' if args.keep_tris else 'removed'}  {tris}")


if __name__ == "__main__":
    parser = ArgumentParser(description="TRUMANS chunk -> G1 mesh 점유/EDT 격자")
    parser.add_argument("--video", required=True, type=str)       # tru_<rec8>_a<NN>_<tag>
    parser.add_argument("--output_root", default="out_trumans", type=str)
    parser.add_argument("--recon_root",
                        default=path.join(CINEMATRAJ_ROOT, "out", "trumans_recon"), type=str)
    parser.add_argument("--out", default="", type=str)            # 비면 <root>/<video>/mesh_grid.npz
    parser.add_argument("--blender", default=BLENDER, type=str)
    parser.add_argument("--voxel", default=0.05, type=float)      # 격자 한 칸 (m)
    parser.add_argument("--pad", default=1.0, type=float)         # AABB 여유 (m)
    parser.add_argument("--clip", default=3.0, type=float)        # 거리 상한 (m, float16 정밀도)
    parser.add_argument("--move_eps", default=5e-4, type=float)   # 정적/동적 경계 (m)
    # 삼각형 원본을 남긴다 (디버깅·재격자화용). 기본은 격자만 남기고 지운다.
    parser.add_argument("--keep_tris", action="store_true", default=False)
    parser.add_argument("--no_keep_tris", dest="keep_tris", action="store_false")
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    main(parser.parse_args())
