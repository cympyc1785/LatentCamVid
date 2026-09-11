"""DIRECTOR(E.T.) 가 낸 카메라를 우리 4D 점군에 렌더한다 — RGB + depth, 소스 영상과 같이.

왜 필요한가: `run_director_vista.py` 는 E.T. 회전 규약을 "미검증"으로 남기고 **위치만** 되돌려
저장했다 (`meta.json:unverified`). 궤적 plot(`pilot_camtraj.mp4`)으로는 카메라가 어디를 보는지
알 수 없다. 렌더가 규약 판정기다 — 회전을 잘못 잡으면 subject 가 화면에 없거나 상하가 뒤집힌다.

## E.T. -> OpenCV c2w

`R_et_w = [e_x; e_y; e_z]` 는 **world 벡터 -> E.T. 성분** 이므로 `v_world = R_et_w.T @ v_et`.
world 에서 본 카메라 축은 `A = R_et_w.T @ R_et[:3,:3]`, 열은 `[c0|c1|c2]`.
E.T. 카메라는 **y-up / z-forward, det=+1** (실측: `col2` 가 char 방향 cos 0.899, `col1` 이
world +Y 와 cos 0.988). OpenCV c2w 는 `[right|down|forward]` 이므로

    down = -c1,   fwd = c2,   right = down x fwd = -c0        # det(R_cv) = +1

= E.T. 프레임을 자기 시선축으로 180° 돌린 것. y-up -> y-down 이 요구하는 것이 정확히 이것이고
다른 선택지는 없다 (좌우가 뒤집힌 `[+c0|-c1|c2]` 는 det = -1 이라 회전이 아니다).

## G frame -> world  (FIX 2026-08-27)

npz 의 `cam_t_g` / `char_g` 는 scene-graph **G frame**(up=+z, 스케일 1/S) 이다. v1 은 이걸
world 로 착각해 렌더러에 그대로 넘겼고, 그 결과 subject 가 49프레임 전부 카메라 뒤(z<0)에
떨어졌다. `g_to_world` 로 `T_wg` 를 먹여야 `cloud.npz` 의 `points_world` 와 같은 게이지가 된다
(회전엔 `T_wg[:3,:3]/S`, 위치엔 `T_wg` 전체 — 회전부에 u->DA3 스케일이 섞여 있다).

`--rot lookat` 이면 E.T. 회전을 **안 쓰고** 위치만 취해 subject track 을 바라보게 만든다 —
회전 규약이 어긋났을 때의 대조군이다 (위치가 맞으면 이쪽은 반드시 제대로 나온다).

## 나오는 것

    <result_dir>/render/<tag>_rgb.mp4      샘플별 splatting RGB (구멍 마젠타)
    <result_dir>/render/<tag>_depth.mp4    샘플별 depth 컬러맵 (turbo, 역깊이 눈금 공유)
    <result_dir>/render/concat.mp4         윗줄 = 소스 + 샘플 RGB / 아랫줄 = 소스 pose depth + 샘플 depth
    <result_dir>/render/index.json         샘플별 hole_fraction / depth 분위수 + 실행 설정

depth 눈금은 **전 샘플 공통**으로 한 번 잡는다. 샘플마다 다시 잡으면 "카메라가 더 가까이 갔다"가
색으로 안 보여서 비교가 성립하지 않는다 (`render_bank_videos.colorize_depth` 와 같은 이유).

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=0 python scripts/render_director_depth.py \
        --result_dir results/20260827_director_camel --video camel
    CUDA_VISIBLE_DEVICES=0 python scripts/render_director_depth.py \
        --result_dir results/20260827_director_camel --video camel \
        --subdirs ab_L ab_R --samples 0 1 --rot lookat
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.overlay import contact_sheet, label_tile, paint_holes                  # noqa: E402
from lbm.render import add_cloud_source_args, look_at_c2w, open_renderer        # noqa: E402
from scripts.render_bank_videos import colorize_depth, write_video              # noqa: E402


def et_to_opencv_c2w(poses_et: np.ndarray, R_et_w: np.ndarray, t_world_u: np.ndarray):
    """(F,4,4) E.T. c2w + (3,3) R_et_w + (F,3) world 위치 -> (F,4,4) OpenCV world c2w.

    회전만 E.T. 에서 가져오고 위치는 이미 되돌려 둔 `cam_t_g` 를 쓴다 (docstring 참조).
    """
    A = np.einsum("ij,fjk->fik", R_et_w.T, poses_et[:, :3, :3].astype(np.float64))
    c0, c1, c2 = A[:, :, 0], A[:, :, 1], A[:, :, 2]
    c2w = np.zeros((len(A), 4, 4), dtype=np.float64)
    c2w[:, :3, 0], c2w[:, :3, 1], c2w[:, :3, 2] = -c0, -c1, c2   # right / down / forward
    c2w[:, :3, 3] = t_world_u
    c2w[:, 3, 3] = 1.0
    det = np.linalg.det(c2w[:, :3, :3])
    assert np.abs(det - 1.0).max() < 1e-4, f"det(R) != 1 (max |det-1| {np.abs(det - 1).max():.2e})"
    return c2w


def lookat_c2w(t_world_u: np.ndarray, char_world_u: np.ndarray, up_world: np.ndarray,
               look_at_bias: float = 0.0, height_u: float = 0.0):
    """E.T. 회전을 버리고 위치만 취해 subject 를 조준한다 — 회전 규약 대조군."""
    up = np.asarray(up_world, dtype=np.float64)
    up = up / np.linalg.norm(up)
    target = char_world_u + look_at_bias * height_u * up
    return np.stack([look_at_c2w(t_world_u[f], target[f], up) for f in range(len(t_world_u))])


def g_to_world(c2w_g: np.ndarray, T_wg: np.ndarray, R_wg: np.ndarray):
    """scene graph G frame c2w -> Vista4D world(DA3) c2w. 위치엔 스케일, 회전엔 R 만."""
    out = np.tile(np.eye(4), (len(c2w_g), 1, 1))
    out[:, :3, :3] = np.einsum("ij,fjk->fik", R_wg, c2w_g[:, :3, :3])
    out[:, :3, 3] = np.einsum("ij,fj->fi", T_wg,
                              np.pad(c2w_g[:, :3, 3], ((0, 0), (0, 1)), constant_values=1.0))[:, :3]
    return out


def render_track(renderer, c2w, height, width):
    """궤적 전 프레임 렌더 -> (rgb 리스트, depth (T,h,w), valid (T,h,w))."""
    rgb, depth, valid = [], [], []
    for f in range(len(c2w)):
        shot = renderer.render(c2w[f], frame=int(f), height=height, width=width)
        good = shot["valid"] & np.isfinite(shot["depth"]) & (shot["depth"] > 0)
        rgb.append(paint_holes(shot["rgb"], shot["valid"]))
        depth.append(shot["depth"].astype(np.float32))
        valid.append(good)
    return rgb, np.stack(depth), np.stack(valid)


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    render_dir = path.join(args.result_dir, args.name)
    makedirs(render_dir, exist_ok=True)

    graph_path = path.join(out_root, args.video, "scene_graph.json")
    with open(graph_path, encoding="utf-8") as file:
        graph = json.load(file)
    # FIX(2026-08-27): npz 의 pose 는 scene graph **G frame** 이다. v1 은 이걸 world 로 착각해
    # 렌더러에 그대로 넘겼다 (subject 가 49프레임 전부 카메라 뒤 z<0 로 떨어졌다).
    T_wg = np.asarray(graph["frames"]["T_wg"], dtype=np.float64)
    scale = float(graph["scale"]["S"])
    R_wg = T_wg[:3, :3] / scale                     # T_wg 의 회전부엔 u->DA3 스케일이 섞여 있다
    assert np.abs(R_wg @ R_wg.T - np.eye(3)).max() < 1e-6, "T_wg 회전부가 직교가 아니다"
    R_gw = np.asarray(graph["frames"]["T_gw"], dtype=np.float64)[:3, :3]
    up_g = R_gw @ np.asarray(graph["gravity"]["up_world"], dtype=np.float64)
    up_g /= np.linalg.norm(up_g)                    # G 에서는 +z

    renderer, recon = open_renderer(args, out_root, graph)
    source_video = recon["video"]

    # 렌더할 궤적 모으기: 최상위 npz + `--subdirs` 의 npz, 각각에서 `--samples` 만.
    folders = [args.result_dir] + [path.join(args.result_dir, s) for s in args.subdirs]
    tracks = []                                     # (tag, c2w (F,4,4))
    char_u = None
    for folder in folders:
        npz_path = path.join(folder, "director_poses.npz")
        if not path.isfile(npz_path):
            print(f"[건너뜀] {npz_path} 없음")
            continue
        npz = np.load(npz_path, allow_pickle=False)
        assert "cam_t_g" in npz, (f"{npz_path} 은 director_pilot_v1 (G/world 혼동 버그) 이다 — "
                                  "run_director_vista.py 로 재생성할 것")
        poses_et = np.asarray(npz["poses_et"], dtype=np.float64)        # (B,F,4,4)
        t_u = np.asarray(npz["cam_t_g"], dtype=np.float64)              # (B,F,3) G frame
        R_et_w = np.asarray(npz["R_et_w"], dtype=np.float64)
        char_u = np.asarray(npz["char_g"], dtype=np.float64)            # (F,3)   G frame
        stem = "s" if folder == args.result_dir else path.basename(folder)
        pick = args.samples if args.samples else list(range(len(poses_et)))
        for b in pick:
            assert b < len(poses_et), f"{npz_path}: sample {b} 없음 (B={len(poses_et)})"
            if args.rot == "et":
                c2w = et_to_opencv_c2w(poses_et[b], R_et_w, t_u[b])
            else:
                c2w = lookat_c2w(t_u[b], char_u, up_g,
                                 args.look_at_bias, args.subject_height_u)
            tracks.append((f"{stem}{b}", g_to_world(c2w, T_wg, R_wg)))
    assert tracks, "렌더할 궤적이 하나도 없다"
    if args.limit > 0:
        tracks = tracks[:args.limit]

    height, width = args.tile_height, args.tile_width
    frames = min(len(c2w) for _, c2w in tracks)

    # 대조군: 소스 카메라 pose 로 렌더한 depth. 눈금의 기준이자 렌더러 자기 일관성 확인이다.
    src_rgb, src_depth, src_valid = render_track(
        renderer, renderer.cam_c2w_src[:frames], height, width)
    src_hole = float(1.0 - src_valid.mean())

    print(f"{'video':<16}{args.video}   rot {args.rot}   tracks {len(tracks)}   frames {frames}")
    print(f"{'source pose':<16}hole {src_hole:.4f}  (0 에 가까워야 한다 — 렌더러 자기 일관성)\n")
    header = f"{'track':<16}{'hole':>8}{'z_p5':>9}{'z_p50':>9}{'z_p95':>9}{'path_da3':>9}"
    print(header)
    print("-" * len(header))

    shots, index = [], {}
    for tag, c2w in tracks:
        rgb, depth, valid = render_track(renderer, c2w[:frames], height, width)
        pool = depth[valid]
        hole = float(1.0 - valid.mean())
        path_u = float(np.linalg.norm(np.diff(c2w[:frames, :3, 3], axis=0), axis=1).sum())
        row = {"hole_fraction": round(hole, 6), "path_len_u": round(path_u, 6),
               "depth_p5": round(float(np.percentile(pool, 5)), 6) if pool.size else None,
               "depth_p50": round(float(np.percentile(pool, 50)), 6) if pool.size else None,
               "depth_p95": round(float(np.percentile(pool, 95)), 6) if pool.size else None}
        index[tag] = row
        shots.append((tag, rgb, depth, valid, row))
        print(f"{tag:<16}{hole:>8.3f}{row['depth_p5']:>9.3f}{row['depth_p50']:>9.3f}"
              f"{row['depth_p95']:>9.3f}{path_u:>9.3f}", flush=True)

    # depth 눈금은 전 트랙 + 소스 pose 를 합쳐 한 번만 (샘플마다 다시 잡으면 비교가 안 된다).
    if args.depth_range:
        lo, hi = float(args.depth_range[0]), float(args.depth_range[1])
    else:
        pool = np.concatenate([src_depth[src_valid].ravel()]
                              + [d[v].ravel() for _, _, d, v, _ in shots])
        lo, hi = (float(x) for x in np.percentile(pool, [5, 95]))
        hi = max(hi, lo * 1.01)
    print(f"\ndepth_range  {lo:.3f} .. {hi:.3f} u  ({'수동' if args.depth_range else '자동 5~95%'})")

    src_depth_rgb = [colorize_depth(src_depth[f], src_valid[f], lo, hi) for f in range(frames)]
    colored = {tag: [colorize_depth(d[f], v[f], lo, hi) for f in range(frames)]
               for tag, _, d, v, _ in shots}

    for tag, rgb, _, _, _ in shots:
        write_video(path.join(render_dir, f"{tag}_rgb.mp4"), rgb, args.fps)
        write_video(path.join(render_dir, f"{tag}_depth.mp4"), colored[tag], args.fps)

    # concat 은 세 가지로 깐다 (`--rows`):
    #   both  윗줄 = 소스 영상 + 샘플 depth warp / 아랫줄 = 소스 pose depth + 샘플 depth 컬러맵
    #   warp  소스 영상 + 샘플 depth warp 만 (기하만 보고 싶을 때는 컬러맵 줄이 방해다)
    #   depth 소스 pose depth + 샘플 depth 컬러맵만
    # `both`/`depth` 에서 아랫줄 첫 칸을 비우지 않고 소스 pose depth 를 넣는 이유 — 기준선이
    # 없으면 구멍이 카메라 탓인지 점군 탓인지 가를 수 없다.
    import cv2
    # `both` 은 반드시 한 줄에 (소스 + 샘플 전부) 가 들어가야 위/아래가 짝이 맞는다.
    columns = len(shots) + 1 if args.rows == "both" else (args.columns or len(shots) + 1)
    sheet = []
    for f in range(frames):
        warp_row = [label_tile(cv2.resize(source_video[f], (width, height)),
                               "SOURCE", f"frame {f}")]
        depth_row = [label_tile(src_depth_rgb[f], "SRC-POSE",
                                f"depth {lo:.2f}..{hi:.2f}u  hole {src_hole:.3f}")]
        for tag, rgb, _, _, row in shots:
            warp_row.append(label_tile(rgb[f], tag,
                                       f"warp  hole {row['hole_fraction']:.3f}  "
                                       f"path {row['path_len_u']:.2f}u  f{f}"))
            depth_row.append(label_tile(colored[tag][f], tag,
                                        f"depth  z50 {row['depth_p50']:.2f}u"))
        tiles = {"both": warp_row + depth_row, "warp": warp_row,
                 "depth": depth_row}[args.rows]
        sheet.append(contact_sheet(tiles, columns))
    output = write_video(path.join(render_dir, args.concat_name), sheet, args.fps)

    meta = {"format": "director_render_v1", "video": args.video,
            "result_dir": path.abspath(args.result_dir), "rot": args.rot,
            "subdirs": args.subdirs, "samples": args.samples, "frames": frames,
            "height": height, "width": width, "fps": args.fps,
            "fixed_focal": args.fixed_focal, "rows": args.rows,
            "depth_range_u": [lo, hi], "source_pose_hole": round(src_hole, 6),
            "tracks": index}
    with open(path.join(render_dir, "index.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=1)

    print(f"\n{'tiles':<16}{len(sheet[0]) and columns} cols x "
          f"{int(np.ceil((len(shots) + 1) * (2 if args.rows == 'both' else 1) / columns))} rows"
          f"   {width}x{height}   rows={args.rows}")
    print(f"-> {output}")
    print(f"-> {render_dir}/index.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)   # CinemaTraj out/ (cloud.npz)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)

    parser.add_argument("--result_dir", required=True, type=str)   # run_director_vista.py 의 --out
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--subdirs", nargs="*", default=[])        # ab_L / cap_static ...
    parser.add_argument("--samples", nargs="*", default=None, type=int)  # 안 주면 전량
    parser.add_argument("--limit", default=0, type=int)            # 타일 상한 (0 = 없음)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--name", default="render", type=str)      # 하위 폴더 이름

    # `et` = E.T. 회전을 OpenCV 로 변환 (규약 판정 대상), `lookat` = 위치만 쓰고 subject 조준.
    parser.add_argument("--rot", default="et", choices=["et", "lookat"])
    parser.add_argument("--look_at_bias", default=0.0, type=float)   # --rot lookat 전용
    parser.add_argument("--subject_height_u", default=0.0, type=float)

    # DA3 focal 떨림을 끄고 frame0 K 로 고정. 소스와 화각을 정확히 맞추려면 off.
    parser.add_argument("--fixed_focal", action="store_true", default=False)
    add_cloud_source_args(parser)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    parser.add_argument("--depth_range", nargs=2, default=None, type=float)
    # concat 에 무엇을 깔지. `warp` = 소스 + depth warp 만 (컬러맵 줄 없음).
    parser.add_argument("--rows", default="both", choices=["both", "warp", "depth"])
    parser.add_argument("--columns", default=0, type=int)      # 0 = 한 줄 (both 은 항상 한 줄)
    parser.add_argument("--concat_name", default="concat.mp4", type=str)
    parser.add_argument("--tile_width", default=480, type=int)
    parser.add_argument("--tile_height", default=270, type=int)
    parser.add_argument("--fps", default=12.0, type=float)
    main(parser.parse_args())
