"""**정적 점군으로 SDF 가 되는가**를 실측한다 (조사용, 파이프라인에 안 걸려 있다).

폐기된 CinemaTraj 계획은 256³ SDF 로 자유공간을 표현하려다 "71% 가 `unknown`" 이라 접었다.
사용자 질문 (2026-08-22): *segmentation 이 있으니 움직이는 물체를 빼고 정적 점군만으로
SDF 가 되는지 보자.* 이게 공짜 질문이 아닌 이유가 있다 — 동적 물체를 빼면 **두 가지**가 달라진다:

  ① 동적 물체의 껍질이 "영구 표면"에서 빠진다 (낙타가 서 있던 자리가 벽이 아니게 된다)
  ② 동적 물체가 **움직이므로**, frame t 에 가려졌던 복셀이 frame t' 에는 뚫린다 — 즉
     multi-frame carving 이 실제로 정보를 **번다**. 단일 프레임 shell 에는 없던 이득이다

②가 얼마나 버는지는 카메라가 아니라 **물체**가 얼마나 움직이는지에 달렸고, 그건 씬마다 다르다.
그래서 추정이 아니라 재야 한다. 세 가지를 같은 격자에서 비교한다:

  A  frame0 한 장          — 폐기된 계획이 서 있던 자리 (single-view shell)
  B  소스 13프레임, 동적 포함 — 카메라 이동만으로 버는 것
  C  소스 13프레임, 동적 제외 — 여기에 segmentation 이 버는 것 (①+②)

판정 기준은 "unknown 이 몇 %냐"가 **아니다**. 격자를 넓게 잡으면 unknown 은 얼마든지 늘어난다.
실제 기준은 두 개다:

  (1) 뱅크 카메라가 실제로 지나가는 자리에서 라벨이 결정되나 (unknown 이면 SDF 로 판정 불가)
  (2) SDF 판정이 G1/G5 판정과 일치하나 — 안 맞으면 게이트를 대체할 수 없다

사용 예시:
    python scripts/probe_static_sdf.py --video camel --grid 256
    python scripts/probe_static_sdf.py --video avocado-slice --grid 256 --device cuda:1
"""
from argparse import ArgumentParser
from os import makedirs, path
import json
import sys

import numpy as np
import torch

HERE = path.dirname(path.dirname(path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from scene_graph.io import load_scene                                          # noqa: E402
from lbm.gates import behind_surface_frames, node_margins, obb_clearance       # noqa: E402

CLOUD_ROOT = HERE
FREE, OCC, UNK_OCCLUDED, UNK_OUTSIDE = 0, 1, 2, 3
LABELS = ("free", "occupied", "unknown(가려짐)", "unknown(시야밖)")


def carve(pts_w, depths, sky, dyn, K, cam_c2w, frames, margin, device,
          static_only: bool, chunk: int = 2_000_000):
    """공간 carving. `pts_w` (M,3) world → 라벨 (M,) int8.

    프레임별 판정은 G1 (`behind_surface_frames`) 과 **같은 식**이다 — `z_cam` 을 그 프레임의
    depth 와 비교한다. 다른 건 점 하나가 아니라 격자 전체를 한 번에 본다는 것뿐이라, 여기서
    나온 라벨과 G1 판정이 어긋나면 그건 구현 차이가 아니라 **격자 해상도** 탓이다.

    `static_only` 면 동적 픽셀은 **표면 투표를 안 한다**. 자유 투표(`z < d − margin`)는 그대로
    유효하다 — 낙타 앞은 프레임 t 에 실제로 비어 있었다. 가림은 어느 모드든 그대로 가림이다
    (낙타 뒤는 그 프레임에서 안 보인다). 이 비대칭이 ②(물체가 비켜서 뚫리는 것)를 만든다.
    """
    height, width = depths.shape[-2:]
    total = len(pts_w)
    free = torch.zeros(total, dtype=torch.bool, device=device)
    surf = torch.zeros(total, dtype=torch.bool, device=device)
    seen = torch.zeros(total, dtype=torch.bool, device=device)        # 어느 프레임에든 유효 관측
    dep_t = torch.as_tensor(np.asarray(depths, dtype=np.float32), device=device)
    sky_t = torch.as_tensor(np.asarray(sky, dtype=bool), device=device)
    dyn_t = torch.as_tensor(np.asarray(dyn, dtype=bool), device=device)
    for start in range(0, total, chunk):
        block = torch.as_tensor(pts_w[start:start + chunk], dtype=torch.float32, device=device)
        for t in frames:
            w2c = torch.as_tensor(np.linalg.inv(cam_c2w[t]), dtype=torch.float32, device=device)
            k = torch.as_tensor(K[t], dtype=torch.float32, device=device)
            cam = block @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2]
            ok = z > 1e-6
            uv = cam @ k.T
            u = torch.where(ok, uv[:, 0] / z.clamp(min=1e-6), torch.full_like(z, -1.0))
            v = torch.where(ok, uv[:, 1] / z.clamp(min=1e-6), torch.full_like(z, -1.0))
            ui, vi = u.floor().long(), v.floor().long()
            inside = ok & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            ui, vi = ui.clamp(0, width - 1), vi.clamp(0, height - 1)
            flat = vi * width + ui
            d = dep_t[t].reshape(-1)[flat]
            is_sky = sky_t[t].reshape(-1)[flat]
            is_dyn = dyn_t[t].reshape(-1)[flat]
            valid = inside & torch.isfinite(d) & (d > 0) & ~is_sky
            sl = slice(start, start + len(block))
            seen[sl] |= valid
            free[sl] |= valid & (z < d - margin)
            vote = valid & ((z - d).abs() <= margin)
            surf[sl] |= (vote & ~is_dyn) if static_only else vote
    label = torch.full((total,), UNK_OCCLUDED, dtype=torch.int8, device=device)
    label[~seen] = UNK_OUTSIDE
    label[surf] = OCC
    label[free] = FREE                       # 자유가 우선 — 표준 carving (얇은 면은 깎인다)
    conflicts = int((free & surf).sum())
    return label.cpu().numpy(), conflicts, seen.cpu().numpy()


def label_at(points_g, origin, step, shape, label_grid):
    """G 좌표 점들 → 그 복셀의 라벨. 격자 밖이면 `UNK_OUTSIDE`."""
    idx = np.floor((np.asarray(points_g, dtype=float) - origin) / step).astype(int)
    good = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=-1)
    out = np.full(len(idx), UNK_OUTSIDE, dtype=np.int8)
    sel = idx[good]
    out[good] = label_grid[sel[:, 0], sel[:, 1], sel[:, 2]]
    return out


def main(args):
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    folder = path.join(out_root, args.video)
    with open(path.join(folder, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)
    recon = load_scene(args.eval_data, args.video, args.vista4d_root,
                       seg_root=args.seg_root, seg_static_root=args.seg_static_root)
    scale = float(graph["scale"]["S"])
    margin = float(args.margin_frac) * scale
    cam_c2w = np.asarray(recon["cam_c2w"], dtype=np.float64)
    K = np.asarray(recon["K"], dtype=np.float64)
    num_frames = len(cam_c2w)
    frames = np.unique(np.linspace(0, num_frames - 1, args.src_frames).round().astype(int)).tolist()
    T_gw = np.asarray(graph["frames"]["T_gw"], dtype=float)
    T_wg = np.linalg.inv(T_gw)

    # ── 격자 범위. **뱅크 카메라가 실제로 가는 데**를 반드시 덮어야 한다 — 안 덮으면
    #    "unknown(시야밖)" 이 격자를 좁게 잡은 내 탓인지 관측이 없어서인지 안 갈린다.
    corners = [np.asarray(n["obb"]["center"], dtype=float) for n in graph["nodes"]]
    corners += [np.asarray(n["obb"]["center"], dtype=float)
                + np.asarray(n["obb"]["extent"], dtype=float).max() for n in graph["nodes"]]
    corners += [np.asarray(n["obb"]["center"], dtype=float)
                - np.asarray(n["obb"]["extent"], dtype=float).max() for n in graph["nodes"]]
    src_g = (T_gw[:3, :3] @ cam_c2w[:, :3, 3].T).T + T_gw[:3, 3]
    corners.append(src_g.min(0))
    corners.append(src_g.max(0))
    bank_poses, bank_ids = None, None
    bank_npz = path.join(folder, "hole_bank", "poses.npz")
    if path.isfile(bank_npz):
        blob = np.load(bank_npz, allow_pickle=True)
        bank_poses = np.asarray(blob["cam_c2w"], dtype=float)          # (V,F,4,4) world
        bank_ids = [str(s) for s in blob["variant_id"]]
        bank_g = (T_gw[:3, :3] @ bank_poses[:, :, :3, 3].reshape(-1, 3).T).T + T_gw[:3, 3]
        corners += [bank_g.min(0), bank_g.max(0)]
    lo = np.min(np.stack(corners), axis=0)
    hi = np.max(np.stack(corners), axis=0)
    pad = args.pad_frac * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    grid = int(args.grid)
    step = float((hi - lo).max() / grid)
    shape = np.maximum(np.ceil((hi - lo) / step).astype(int), 1)

    axes = [lo[i] + (np.arange(shape[i]) + 0.5) * step for i in range(3)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    pts_w = (T_wg[:3, :3] @ mesh.T).T + T_wg[:3, 3]

    # 소스 관측의 **기하학적 천장**. carving 이 무엇이든 말할 수 있는 영역은 소스 frustum 합집합
    # 안뿐이고, 그 합집합의 크기는 (a) 화각 (b) 카메라가 움직인 거리로 정해진다. 둘 다 작으면
    # 자유공간은 가느다란 원뿔 하나고, 그 밖은 동적 물체를 아무리 걷어내도 unknown 이다.
    hfov = 2.0 * np.degrees(np.arctan(float(recon["depths"].shape[-1]) / 2.0 / float(K[0][0, 0])))
    vfov = 2.0 * np.degrees(np.arctan(float(recon["depths"].shape[-2]) / 2.0 / float(K[0][1, 1])))
    track = float(np.linalg.norm(cam_c2w[:, :3, 3] - cam_c2w[0, :3, 3], axis=-1).max())
    print(f"{'video':<16}{args.video}")
    print(f"{'소스 카메라':<16}hfov {hfov:.1f}°  vfov {vfov:.1f}°   이동 최대 {track:.4f} DA3 "
          f"= {track / scale:.4f} u   parallax_ratio {graph['scale']['parallax_ratio']:.4f}")
    print(f"{'격자':<16}{tuple(shape)} = {len(mesh):,} 복셀   step {step:.4f} u  "
          f"({step * scale:.4f} DA3)")
    print(f"{'범위 (u)':<16}x [{lo[0]:+.2f}, {hi[0]:+.2f}]  y [{lo[1]:+.2f}, {hi[1]:+.2f}]  "
          f"z [{lo[2]:+.2f}, {hi[2]:+.2f}]")
    print(f"{'소스 프레임':<16}{frames}   margin {args.margin_frac:g}·S = {margin:.4f} DA3")
    print(f"{'뱅크':<16}" + (f"{len(bank_ids)} 변이 x {bank_poses.shape[1]} 프레임"
                             if bank_poses is not None else "없음 (fit_hole_ladder 먼저)"))
    print()

    variants = [("A frame0 1장", [0], False),
                (f"B {len(frames)}프레임 동적포함", frames, False),
                (f"C {len(frames)}프레임 동적제외", frames, True)]
    results = {}
    # 전체 격자 대비 비율은 **뜻이 없다** — 격자를 넓게 잡으면 unknown 은 얼마든지 커진다.
    # 그래서 `frustum 안` (= 어느 소스 프레임엔가 유효하게 관측된 복셀) 을 분모로 따로 낸다.
    header = (f"{'변이':<22}{'frustum내':>10}{'free|내':>9}{'occ|내':>8}{'unk|내':>8}"
              f"{'충돌':>7}   뱅크 pose  frustum내 / free")
    print(header)
    print("-" * len(header))
    for name, use_frames, static_only in variants:
        label, conflicts, seen = carve(pts_w, recon["depths"], recon["sky_mask"],
                                       recon["dynamic_mask"], K, cam_c2w, use_frames, margin,
                                       args.device, static_only)
        grid_label = label.reshape(tuple(shape))
        inside = max(int(seen.sum()), 1)
        counts = np.bincount(label.astype(int), minlength=4)
        bank_txt = "—"
        if bank_poses is not None:
            pg = (T_gw[:3, :3] @ bank_poses[:, :, :3, 3].reshape(-1, 3).T).T + T_gw[:3, 3]
            bl = label_at(pg, lo, step, shape, grid_label)
            n = len(bl)
            bank_txt = (f"{(bl != UNK_OUTSIDE).sum() / n:.3f} / {(bl == FREE).sum() / n:.3f}")
            results[name] = {"bank_label": bl}
        results.setdefault(name, {}).update(
            {"counts": counts.tolist(), "conflicts": conflicts, "grid": grid_label,
             "inside_frac": inside / len(label)})
        print(f"{name:<22}{inside / len(label):>10.3f}{counts[FREE] / inside:>9.3f}"
              f"{counts[OCC] / inside:>8.3f}{counts[UNK_OCCLUDED] / inside:>8.3f}"
              f"{conflicts:>7}   {bank_txt}")

    # ── SDF. `free` 에서 non-free 까지의 EDT. 이게 플래너가 쓸 값이다.
    from scipy.ndimage import distance_transform_edt
    print()
    sdf_header = (f"{'변이':<22}{'sdf p05':>9}{'p50':>8}{'p95':>8}   "
                  f"뱅크 pose SDF (u)  min / p05 / median")
    print(sdf_header)
    print("-" * len(sdf_header))
    for name, _, _ in variants:
        grid_label = results[name]["grid"]
        free_mask = grid_label == FREE
        # 자유 복셀에서 **자유가 아닌 곳**까지의 거리. unknown 을 장애물로 친다 (보수적) —
        # 이게 폐기된 계획이 하려던 것이고, unknown 이 많으면 여기서 값이 죽는다.
        sdf = distance_transform_edt(free_mask, sampling=step)
        vals = sdf[free_mask]
        bank_txt = "—"
        if bank_poses is not None:
            pg = (T_gw[:3, :3] @ bank_poses[:, :, :3, 3].reshape(-1, 3).T).T + T_gw[:3, 3]
            idx = np.floor((pg - lo) / step).astype(int)
            good = np.all((idx >= 0) & (idx < shape), axis=-1)
            bs = np.zeros(len(idx))
            bs[good] = sdf[idx[good, 0], idx[good, 1], idx[good, 2]]
            results[name]["bank_sdf"] = bs
            bank_txt = (f"{bs.min():.4f} / {np.percentile(bs, 5):.4f} / "
                        f"{np.median(bs):.4f}")
        print(f"{name:<22}{np.percentile(vals, 5):>9.4f}{np.percentile(vals, 50):>8.4f}"
              f"{np.percentile(vals, 95):>8.4f}   {bank_txt}")

    # ── 게이트 대체 가능성. SDF 판정 vs G1 / G5 판정을 뱅크 pose 에서 대조한다.
    if bank_poses is not None:
        flat = bank_poses.reshape(-1, 4, 4)
        sub = np.arange(0, len(flat), max(1, len(flat) // args.check_poses))
        g1 = np.asarray([bool(behind_surface_frames(
            flat[i][:3, 3], recon["depths"], K=K, cam_c2w=cam_c2w,
            sky_mask=recon["sky_mask"], scale=scale, frames=frames,
            margin_frac=args.margin_frac, clear_frac=0.0, radius_px=2)) for i in sub])
        margins = node_margins(graph["nodes"], 0.0, args.obb_floor, 0.12)
        # `flat` 은 변이를 이어 붙인 것이라 인덱스가 프레임 번호가 아니다 — 동적 노드 OBB 를
        # 프레임별로 꺼내려면 **변이 안에서의 프레임 번호**를 따로 줘야 한다 (안 주면 49 에서 터진다).
        _, _, slacks = obb_clearance(flat[sub], graph["nodes"], T_gw,
                                     frames=(sub % bank_poses.shape[1]).tolist(), margins=margins)
        g5 = np.asarray(slacks) < 0.0
        print(f"\n{'대조 표본':<16}{len(sub)} pose   G1 위반 {int(g1.sum())}   "
              f"G5 위반 {int(g5.sum())}")
        head = (f"{'변이':<22}{'SDF=free':>10}{'SDF<=0':>9}{'G1 일치':>9}{'G1 놓침':>9}"
                f"{'G5 일치':>9}{'G5 놓침':>9}")
        print(head)
        print("-" * len(head))
        for name, _, _ in variants:
            bl = results[name]["bank_label"][sub]
            bs = results[name]["bank_sdf"][sub]
            sdf_bad = bl != FREE                    # SDF 가 "여기 두면 안 된다" 라고 하는 것
            print(f"{name:<22}{(bl == FREE).sum() / len(sub):>10.3f}"
                  f"{sdf_bad.sum() / len(sub):>9.3f}"
                  f"{(sdf_bad & g1).sum() / max(g1.sum(), 1):>9.3f}"
                  f"{(~sdf_bad & g1).sum() / max(g1.sum(), 1):>9.3f}"
                  f"{(sdf_bad & g5).sum() / max(g5.sum(), 1):>9.3f}"
                  f"{(~sdf_bad & g5).sum() / max(g5.sum(), 1):>9.3f}")
            _ = bs

    if args.save:
        makedirs(path.join(folder, "sdf_probe"), exist_ok=True)
        target = path.join(folder, "sdf_probe", "probe.npz")
        np.savez_compressed(
            target, origin=lo, step=step, shape=shape, T_gw=T_gw, frames=np.asarray(frames),
            **{f"label_{i}": results[n]["grid"] for i, (n, _, _) in enumerate(variants)})
        print(f"\n-> {target}")


if __name__ == "__main__":
    parser = ArgumentParser(description="정적 점군 SDF 타당성 실측 (조사용)")
    parser.add_argument("--video", default="camel", type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--eval_data", default="/data1/cympyc1785/data/Vista4D-Eval-Data", type=str)
    parser.add_argument("--vista4d_root",
                        default="/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D",
                        type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    parser.add_argument("--grid", default=256, type=int)          # 최장축 기준 복셀 수
    parser.add_argument("--pad_frac", default=0.1, type=float)    # 격자 여백 (최장축 대비)
    parser.add_argument("--src_frames", default=13, type=int)     # G1 과 같은 기본값
    parser.add_argument("--margin_frac", default=0.02, type=float)
    parser.add_argument("--obb_floor", default=0.1532, type=float)   # camel 실효 마진
    parser.add_argument("--check_poses", default=2000, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--no_save", dest="save", action="store_false")
    main(parser.parse_args())
