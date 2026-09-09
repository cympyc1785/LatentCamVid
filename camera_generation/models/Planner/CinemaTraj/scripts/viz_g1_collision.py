"""G1 (behind-surface) 판정을 **그림으로 보여준다** — 어느 소스 프레임이, 어느 픽셀에서,
얼마나 차이로 "카메라가 표면 뒤"라고 말했는지.

왜 필요한가 (사용자 질문 D112): "G1 충돌 어떻게 판정된건지 시각화해서 보여줘."
뱅크 CSV 는 `behind_frames` / `behind_frac` / `behind_worst_src` 세 숫자만 남긴다. 그런데
`binding=collision` 이 전체의 7.7%, `clamped_low` 안에서는 46.4% 라 **첫 벽**으로 제일 자주
나오는 게이트인데, 숫자만으로는 "정말 벽 속인가 / 재구성 잡음인가"를 못 가른다.

## 판정 자체 (`lbm/gates.py:46 behind_surface_frames`)

렌더가 없다. 플랜 카메라의 **위치 한 점** `p` 를 소스 프레임 t 의 카메라로 되쏘아:

    z_cam(p) + clear_frac·S  >  depth_t(u,v) + margin_frac·S      →  프레임 t 는 "hit"

즉 "그 소스 카메라에서 봤을 때 p 가 관측된 표면보다 **멀다**" = 표면 뒤. 하늘·무효 depth 는
판정 대상이 아니다 (하늘 뒤는 벽 뒤가 아니다). `radius_px=2` 면 투영 픽셀 주변 5×5 의 **최소**
depth 를 쓴다 — 얇은 물체 가장자리를 스칠 때 옆 배경의 먼 depth 가 잡히는 걸 막는다.
`clear_frac=0.10` 은 표면 앞에 **요구하는** 여유라서, 실제 관통이 아니라 "너무 붙었다"도
hit 이 된다. 이 그림에서 그 둘을 색으로 구분한다 (관통 = 빨강, 여유 부족 = 주황).

한 플랜 프레임이 hit 소스 프레임을 **하나라도** 가지면 그 프레임은 위반이고,
`behind_frac` = 위반 프레임 수 / 49 다.

## 그림 (프레임당 1장, 49장 → mp4)

  왼쪽  플랜 카메라가 그 프레임에서 실제로 보는 렌더 (구멍은 마젠타). 벽 속이면 여기가 무너진다.
  오른쪽 소스 probe 프레임 타일 격자. 각 타일에 `p` 의 투영점을 찍고 테두리 색으로 판정을
        표시한다. 타일 안 글씨는 `z_cam` 대 `depth` (둘 다 u 단위).

색: 빨강 = 관통(z > depth+margin), 주황 = 여유 부족(clear 때문에 걸림), 초록 = 통과,
회색 = 판정 불가(화면 밖 / 카메라 뒤 / 하늘 / 무효 depth).

env: `vista4d` (렌더러 때문). `--no_render` 면 오른쪽 격자만 만들고 GPU 가 필요 없다.

예시:
    CUDA_VISIBLE_DEVICES=2 python scripts/viz_g1_collision.py \
        --video parkour --bank_dir hole_bank_d110 --output_root out_dynpose \
        --variant dyn_0__dolly_in__hole0.5 --out /tmp/g1_parkour.mp4
    # 그 뱅크에서 behind_frac 이 가장 높은 변이를 자동 선택
    CUDA_VISIBLE_DEVICES=2 python scripts/viz_g1_collision.py --video parkour --worst
"""
import csv
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import cv2
import imageio.v2 as imageio
import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.render import CloudRenderer                                            # noqa: E402
from scene_graph.io import load_scene                                           # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402

# 판정 결과 → (BGR 색, 라벨). `gates.behind_surface_frames` 의 분기와 1:1 이다.
VERDICT = {"pierce": ((0, 0, 235), "PIERCE"),      # z > depth + margin — 진짜 표면 뒤
           "tight": ((0, 140, 255), "TIGHT"),      # clear_frac 때문에 걸림 — 표면에 너무 붙음
           "clear": ((0, 190, 0), "clear"),
           "skip": ((150, 150, 150), "n/a")}       # 화면 밖 / 카메라 뒤 / 하늘 / 무효 depth


def judge_frame(p_world, t, depths, K, cam_c2w, sky_mask, scale, margin_frac, clear_frac,
                radius_px):
    """소스 프레임 t 하나에 대한 G1 판정. `(verdict, uv, z_cam, depth)`.

    `gates.behind_surface_frames` 를 **복제**한 게 아니라 같은 식을 쓰되 중간값을 돌려준다 —
    거기서는 hit 프레임 인덱스만 나와서 그릴 게 없다. 식이 바뀌면 두 곳을 같이 고쳐야 한다.
    """
    height, width = depths.shape[-2:]
    margin, clear = margin_frac * scale, clear_frac * scale
    w2c = np.linalg.inv(cam_c2w[t])
    cam = w2c[:3, :3] @ np.asarray(p_world, float) + w2c[:3, 3]
    if cam[2] <= 1e-6:
        return "skip", None, float(cam[2]), float("nan")
    uv = (K[t] @ cam)[:2] / cam[2]
    u, v = int(np.floor(uv[0])), int(np.floor(uv[1]))
    if not (0 <= u < width and 0 <= v < height):
        return "skip", uv, float(cam[2]), float("nan")
    if radius_px <= 0:
        if sky_mask[t][v, u]:
            return "skip", uv, float(cam[2]), float("nan")
        z = float(depths[t][v, u])
        if not (np.isfinite(z) and z > 0):
            return "skip", uv, float(cam[2]), float("nan")
    else:
        u0, u1 = max(u - radius_px, 0), min(u + radius_px + 1, width)
        v0, v1 = max(v - radius_px, 0), min(v + radius_px + 1, height)
        patch = depths[t][v0:v1, u0:u1]
        usable = np.isfinite(patch) & (patch > 0) & ~sky_mask[t][v0:v1, u0:u1]
        if not usable.any():
            return "skip", uv, float(cam[2]), float("nan")
        z = float(patch[usable].min())
    if float(cam[2]) + clear <= z + margin:
        return "clear", uv, float(cam[2]), z
    # 걸리긴 걸렸다. clear 여유가 없었어도 걸렸다면 진짜 관통이다.
    return ("pierce" if float(cam[2]) > z + margin else "tight"), uv, float(cam[2]), z


def put(img, text, org, color=(255, 255, 255), scale=0.42, thick=1):
    """검은 테두리를 깔고 글씨 — 밝은 배경에서도 읽히게."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    bank_folder = path.join(out_root, args.video, args.bank_dir)
    with open(path.join(bank_folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    variants = bank["variants"]
    cam_c2w_all = np.load(path.join(bank_folder, "poses.npz"), allow_pickle=False)["cam_c2w"]
    assert len(cam_c2w_all) == len(variants), \
        f"poses.npz {len(cam_c2w_all)} != bank.json {len(variants)} — 뱅크를 다시 만들 것"

    rows = {}
    csv_path = path.join(bank_folder, "bank.csv")
    if path.exists(csv_path):
        with open(csv_path, newline="") as file:
            rows = {r["variant_id"]: r for r in csv.DictReader(file)}

    if args.worst:
        # behind_frac 이 가장 큰 변이. 뭘 볼지 모를 때의 기본 진입점이다.
        scored = [(float(r.get("behind_frac") or 0.0), vid) for vid, r in rows.items()]
        assert scored, "bank.csv 가 없어서 --worst 를 못 쓴다. --variant 로 지정할 것"
        args.variant = max(scored)[1]
    assert args.variant, "--variant 또는 --worst 중 하나는 필요하다"
    index = next((i for i, v in enumerate(variants) if v["variant_id"] == args.variant), None)
    assert index is not None, f"변이를 못 찾았다: {args.variant}"
    poses = np.asarray(cam_c2w_all[index], dtype=np.float64)
    row = rows.get(args.variant, {})

    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                       seg_root=args.seg_root, seg_static_root=args.seg_static_root)
    scale = float(graph["scale"]["S"])
    depths, sky_mask = recon["depths"], recon["sky_mask"]

    renderer = None
    if args.render:
        renderer = CloudRenderer(path.join(out_root, args.video, "cloud.npz"),
                                 vista4d_root=args.vista4d_root, device=args.device,
                                 fixed_focal=args.fixed_focal)
    K_src = (renderer.K_src if renderer is not None
             else np.asarray(recon["K"], dtype=np.float64))
    cam_c2w_src = (renderer.cam_c2w_src if renderer is not None
                   else np.asarray(recon["cam_c2w"], dtype=np.float64))
    num_src = len(depths)
    # fit 이 실제로 쓴 probe 집합과 **같은 방식**으로 뽑는다 (`sample_camera_bank.behind_context`).
    probe = np.unique(np.linspace(0, num_src - 1, args.behind_src_frames)
                      .round().astype(int)).tolist()

    src_h, src_w = depths.shape[-2:]
    tile_w = args.tile_width
    tile_h = int(round(tile_w * src_h / src_w))
    cols = args.tile_cols
    grid_rows = int(np.ceil(len(probe) / cols))
    grid_w, grid_h = cols * tile_w, grid_rows * tile_h

    # 왼쪽 렌더 패널은 격자와 높이를 맞춘다 (렌더 없으면 폭 0).
    pan_h = grid_h
    pan_w = int(round(pan_h * src_w / src_h)) if renderer is not None else 0
    head = 46
    canvas_w, canvas_h = pan_w + grid_w, head + grid_h

    images = np.asarray(recon["video"])           # (F,H,W,3) uint8 RGB
    frames_out = []
    behind_count = 0
    # 판정 사유 누적. pierce 가 0 이고 tight 만 잔뜩이면 "벽 속"이 아니라 `clear_frac` 이
    # 요구한 여유가 없는 것뿐이다 — 게이트를 풀지 손잡이를 줄일지가 여기서 갈린다.
    tally = {"pierce": 0, "tight": 0, "clear": 0, "skip": 0}
    pierce_frames = 0
    for f in range(len(poses)):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        p = poses[f][:3, 3]
        verdicts = []
        for j, t in enumerate(probe):
            verdict, uv, z_cam, z_surf = judge_frame(
                p, int(t), depths, K_src, cam_c2w_src, sky_mask, scale,
                args.behind_margin_frac, args.behind_clear_frac, args.behind_radius_px)
            verdicts.append(verdict)
            color, label = VERDICT[verdict]
            tile = cv2.resize(images[int(t)][:, :, ::-1], (tile_w, tile_h))   # RGB→BGR
            tile = (tile * 0.72).astype(np.uint8)                             # 마커가 튀게 어둡게
            if uv is not None:
                cu = int(round(uv[0] * tile_w / src_w))
                cv_ = int(round(uv[1] * tile_h / src_h))
                if 0 <= cu < tile_w and 0 <= cv_ < tile_h:
                    cv2.drawMarker(tile, (cu, cv_), color, cv2.MARKER_CROSS, 15, 2)
                    cv2.circle(tile, (cu, cv_), 7, color, 2)
                    if args.behind_radius_px > 0:      # 실제로 본 패치 (5x5 를 타일 배율로)
                        r = max(int(round(args.behind_radius_px * tile_w / src_w)), 1)
                        cv2.rectangle(tile, (cu - r, cv_ - r), (cu + r, cv_ + r), color, 1)
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), color,
                          3 if verdict in ("pierce", "tight") else 1)
            put(tile, f"src {t:>2}  {label}", (5, 14), color)
            if np.isfinite(z_surf):
                put(tile, f"z {z_cam / scale:.3f}  surf {z_surf / scale:.3f}", (5, tile_h - 6))
            gy, gx = divmod(j, cols)
            canvas[head + gy * tile_h:head + (gy + 1) * tile_h,
                   pan_w + gx * tile_w:pan_w + (gx + 1) * tile_w] = tile

        hits = sum(1 for v in verdicts if v in ("pierce", "tight"))
        behind_count += 1 if hits else 0
        for v in verdicts:
            tally[v] += 1
        pierce_frames += 1 if any(v == "pierce" for v in verdicts) else 0
        if renderer is not None:
            out = renderer.render(poses[f], frame=min(int(f), num_src - 1),
                                  height=pan_h, width=pan_w)
            rgb = out["rgb"].copy()
            rgb[~out["valid"]] = (255, 0, 255)                                 # 구멍 = 마젠타
            canvas[head:head + pan_h, 0:pan_w] = rgb[:, :, ::-1]
            put(canvas, f"plan camera view  (hole {1.0 - float(out['valid'].mean()):.2f})",
                (8, head + 20), (255, 255, 255), 0.5)
            cv2.rectangle(canvas, (0, head), (pan_w - 1, head + pan_h - 1),
                          VERDICT["pierce" if hits else "clear"][0], 3)

        bar = VERDICT["pierce"][0] if hits else VERDICT["clear"][0]
        cv2.rectangle(canvas, (0, 0), (canvas_w, head - 1), (24, 24, 24), -1)
        put(canvas, f"{args.video}  {args.variant}", (8, 18), (255, 255, 255), 0.5)
        put(canvas, f"frame {f:>2}/{len(poses) - 1}   G1 hits {hits}/{len(probe)}"
                    f"   behind so far {behind_count}/{f + 1}"
                    f"   [bank behind_frac {row.get('behind_frac', '?')}"
                    f"  binding {row.get('binding', '?')}  status {row.get('status', '?')}]",
            (8, 38), bar, 0.46)
        frames_out.append(canvas[:, :, ::-1])                                  # BGR→RGB

    out_path = args.out or path.join(bank_folder, f"g1_{args.variant}.mp4")
    makedirs(path.dirname(path.abspath(out_path)), exist_ok=True)
    imageio.mimwrite(out_path, frames_out, fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"{'video':<18}{args.video}")
    print(f"{'variant':<18}{args.variant}")
    print(f"{'probe frames':<18}{probe}")
    print(f"{'margin/clear/rad':<18}{args.behind_margin_frac} / {args.behind_clear_frac} / "
          f"{args.behind_radius_px} px   (S={scale:.4f})")
    print(f"{'behind frames':<18}{behind_count}/{len(poses)} = {behind_count / len(poses):.4f}"
          f"   (bank behind_frac {row.get('behind_frac', '?')})")
    print(f"{'  그중 pierce 포함':<18}{pierce_frames}/{len(poses)}"
          f"   ← 0 이면 전부 '표면에 너무 붙음'(clear_frac)이지 관통이 아니다")
    tot = sum(tally.values())
    print(f"{'판정 (프레임×소스)':<18}"
          + "   ".join(f"{k} {v}({100.0 * v / tot:.1f}%)" for k, v in tally.items()))
    print(f"{'binding/status':<18}{row.get('binding', '?')} / {row.get('status', '?')}")
    print(f"{'out':<18}{out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="G1 behind-surface 판정 시각화")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_root", default=None)          # 비우면 CinemaTraj/out
    parser.add_argument("--bank_dir", default="hole_bank")
    parser.add_argument("--variant", default=None)              # 예: dyn_0__dolly_in__hole0.5
    parser.add_argument("--worst", action="store_true")         # behind_frac 최대 변이 자동 선택
    parser.add_argument("--out", default=None)                  # 비우면 뱅크 폴더 안에
    # G1 손잡이 — fit 기본값과 **같게** 두어야 뱅크 판정을 재현한다 (fit_hole_ladder.py:1087-1092)
    parser.add_argument("--behind_src_frames", default=13, type=int)
    parser.add_argument("--behind_margin_frac", default=0.02, type=float)
    parser.add_argument("--behind_clear_frac", default=0.10, type=float)
    parser.add_argument("--behind_radius_px", default=2, type=int)
    # 왼쪽 렌더 패널. 끄면 GPU 없이 격자만 (판정 자체는 렌더가 필요 없다)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no_render", dest="render", action="store_false")
    parser.add_argument("--tile_width", default=214, type=int)
    parser.add_argument("--tile_cols", default=5, type=int)
    parser.add_argument("--fps", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    #    **기본 True.** 뱅크 pose 는 frame0 고정 K 로 풀렸다. 프레임별 DA3 K 로 그리면 화각이
    #    떨려(snowboard fx 진폭 6.99%) 충돌 그림이 궤적과 무관하게 흔들린다. 게다가 이 스크립트는
    #    `renderer.K_src` 를 G1 판정에도 쓰므로 규약이 어긋나면 판정 자체가 달라진다.
    #    `--no_fixed_focal` 은 예전 동작(프레임별 K)이다.
    parser.add_argument("--fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--seg_root", default=None)
    parser.add_argument("--seg_static_root", default=None)
    main(parser.parse_args())
