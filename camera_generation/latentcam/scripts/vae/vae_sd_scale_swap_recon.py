"""[new] Scene-Decoupled cross-clip: **context clip 의 avg_scale_align 을 target 의 분모로** 쓸 때
frozen CameraVAE 가 버티는가?

배경 / 왜 이게 문제인가
----------------------
계획은 같은 scene 의 clip 하나를 context, 다른 하나를 target 으로 잡고 target 의 cam_param 을
**context clip 의 avg_scale_align** 으로 정규화하는 것이다. 그런데 실측(300 scene, 5328 pair):

  avg_scale_align == s * avg_scale  가 **정확히** 성립한다 (상대오차 max 0.0).
  -> 즉 자기 clip 의 align 으로 나누면 sim3 이전(native avg_scale)과 **비트 단위로 같은**
     cam_param 이 나온다. sim3 가 실제로 무언가를 바꾸는 유일한 지점이 이 cross-clip 분모다.

  own/ctx 비 : p1 0.228  p10 0.709  p50 1.000  p90 1.410  p99 4.389  min 0.059  max 16.921

cam_param 의 translation 3채널은 분모에 반비례하므로, 이 비가 그대로 VAE 입력 크기의 배율이
된다. VAE(`vae_20260302_300.pth`)는 자기 clip 분모로 정규화된 분포에서 학습된 frozen 모델이라
16배 큰 입력은 학습 분포 밖이다. 여기서 재는 것:

  in_c_norm       입력 translation 크기 mean ||t||       (own vs swap)
  lat_std_raw     encode 직후 latent std                  -> vae_latent_scale 재측정 필요 여부
  rec_trans       translation 3채널 recon MAE (정규화 단위)
  rec_trans_m     위 * (그 샘플에 쓴 분모, meters)        -> **미터 단위 실오차**
  rec_rot6d       회전 6채널 MAE (분모와 무관해야 정상)
  rec_intr        intr 2채널 MAE (intr_norm='rel' 이라 입력이 전부 [1,1])

ratio 버킷별로도 쪼개서, 어느 배율부터 무너지는지 본다.

cam_param 구성은 dataset_dl3dv.py 와 동일한 규약을 따른다:
  rot6d = normalized_extrinsics[:, :3, 0] ++ [:, :3, 1]  (w2c R 의 앞 두 열)
  trans = normalized_extrinsics[:, :3, 3]                (trans_repr='w2c' 기본)
  intr  = [fx/(2cx), fy/(2cy)] 를 frame0 으로 나눔       (intr_norm='rel')
da3 pose.npz 의 extrinsics 는 da3 자체 단위라, GT meters 로 올리려면 translation 에 sim3 의
스칼라 s 를 곱한다 (회전 R 과 평행이동 t 는 상대 extrinsic 에 영향이 없어 곱할 필요가 없다).

usage:
  python scripts/vae/vae_sd_scale_swap_recon.py --n-scenes 400 --split whuman
env/args: --exp (cfg 로드용 experiment, 기본 da3_1k_da3pose) --device cuda:N
out -> results/scene_decoupled/vae_scale_swap.json
"""
import os, sys, json, random, argparse
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg                                    # noqa: E402
from utils.data_utils import normalize_camera_extrinsics_and_points  # noqa: E402
from models.vae_intr_large import CameraVAE                       # noqa: E402


def load_clip(split, scene, clip_dir, num_frames):
    """da3 clip -> (ext (T,4,4) w2c da3 단위, K (T,3,3), meta dict) 또는 None (static/결측)."""
    d = os.path.join(ROOT, "da3", split, scene, clip_dir)
    u = os.path.join(d, "umeyama_gt.json")
    if not os.path.isfile(u):
        return None
    j = json.load(open(u))
    if not j.get("moving"):
        return None                                   # static 은 sim3/avg_scale_align 자체가 없다
    ag = os.path.join(d, "avg_scale_align", "0.json")
    if not os.path.isfile(ag):
        return None
    z = np.load(os.path.join(d, "pose.npz"))
    ext34 = z["extrinsics"][:num_frames].astype(np.float64)
    K = z["intrinsics"][:num_frames].astype(np.float64)
    if len(ext34) < num_frames:
        return None
    ext = np.tile(np.eye(4), (num_frames, 1, 1))
    ext[:, :3, :4] = ext34
    meta = {"s": float(j["s"]), "resid": float(j["resid_rmse_over_rad"]),
            "align": float(json.load(open(ag))),
            "avg": float(json.load(open(os.path.join(d, "avg_scale", "0.json"))))}
    return torch.from_numpy(ext).float(), torch.from_numpy(K).float(), meta


def cam_param(ext, K, s_tgt, divisor_m, cfg):
    """dataset_dl3dv.py 와 같은 (T,11). divisor_m 은 **GT meters** 단위 분모."""
    e = ext.clone()
    e[:, :3, 3] *= s_tgt                              # da3 단위 -> GT meters
    div = torch.tensor([divisor_m], dtype=torch.float32)
    ne, _, _, _ = normalize_camera_extrinsics_and_points(
        e, avg_scale=div, max_trans_norm=getattr(cfg, 'max_trans_norm', False))
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    wv, hv = K[:, 0, 2] * 2, K[:, 1, 2] * 2
    ni = torch.stack([fx / wv, fy / hv], dim=-1)
    ni = ni / ni[0:1, :]                              # intr_norm='rel'
    return torch.cat([ne[:, :3, 0], ne[:, :3, 1], ne[:, :3, 3], ni], dim=-1).float()


def report(name, x, xr, divs):
    d = (xr - x).abs()
    rt = d[..., 6:9].mean(dim=(1, 2))                 # (N,) 샘플별
    return {
        "arm": name, "n": int(x.shape[0]),
        "in_c_norm": float(x[..., 6:9].norm(dim=-1).mean()),
        "in_trans_std": float(x[..., 6:9].std()),
        "divisor_mean_m": float(divs.mean()),
        "rec_trans": float(rt.mean()),
        "rec_trans_m": float((rt * divs).mean()),
        "rec_rot6d": float(d[..., 0:6].mean()),
        "rec_intr": float(d[..., 9:11].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="whuman")
    ap.add_argument("--n-scenes", type=int, default=400)
    ap.add_argument("--pairs-per-scene", type=int, default=2)
    ap.add_argument("--num-frames", type=int, default=49)
    ap.add_argument("--exp", default="da3_1k_da3pose", help="frozen VAE 경로/규약을 가져올 experiment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, _ = load_cfg("config", overrides=[f"experiment={args.exp}"])
    print(f"[cfg] vae_ckpt={cfg.vae_ckpt_path}  vae_latent_scale={cfg.vae_latent_scale} "
          f"max_trans_norm={getattr(cfg,'max_trans_norm',False)} cam_dim={cfg.cam_dim}")
    vae = CameraVAE(latent_dim=cfg.cam_dim).to(args.device)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=args.device))
    vae.eval()

    sd = os.path.join(ROOT, "da3", args.split)
    scenes = sorted(s for s in os.listdir(sd) if os.path.isdir(os.path.join(sd, s)))
    rng = random.Random(args.seed); rng.shuffle(scenes)

    X_own, X_swp, D_own, D_swp, RATIO, RESID = [], [], [], [], [], []
    used = 0
    for sc in scenes:
        if used >= args.n_scenes:
            break
        cdir = os.path.join(sd, sc)
        clips = sorted(c for c in os.listdir(cdir)
                       if os.path.isfile(os.path.join(cdir, c, "pose.npz")))
        loaded = [(c, load_clip(args.split, sc, c, args.num_frames)) for c in clips]
        loaded = [(c, v) for c, v in loaded if v is not None]
        if len(loaded) < 2:
            continue
        used += 1
        for _ in range(args.pairs_per_scene):
            (_, (et, Kt, mt)), (_, (_, _, mc)) = rng.sample(loaded, 2)   # target, context
            X_own.append(cam_param(et, Kt, mt["s"], mt["align"], cfg))
            X_swp.append(cam_param(et, Kt, mt["s"], mc["align"], cfg))
            D_own.append(mt["align"]); D_swp.append(mc["align"])
            RATIO.append(mt["align"] / mc["align"]); RESID.append(mt["resid"])

    x_own = torch.stack(X_own); x_swp = torch.stack(X_swp)
    d_own = torch.tensor(D_own); d_swp = torch.tensor(D_swp)
    ratio = torch.tensor(RATIO)
    print(f"\nscenes used={used}  pairs={x_own.shape[0]}  T={x_own.shape[1]}")

    out = {"vae_ckpt": str(cfg.vae_ckpt_path), "vae_latent_scale": float(cfg.vae_latent_scale),
           "n_scenes": used, "n_pairs": int(x_own.shape[0]), "num_frames": args.num_frames,
           "ratio_pct": {str(p): float(np.percentile(np.array(RATIO), p))
                         for p in (1, 10, 50, 90, 99)},
           "arms": [], "ratio_buckets": []}

    lat = {}
    for name, x, dv in (("own(target 자기 align)", x_own, d_own),
                        ("swap(context align)", x_swp, d_swp)):
        with torch.no_grad():
            z = vae.encode(x.to(args.device))
            if not torch.is_tensor(z):
                z = z[0]
            xr = vae.decode(z).cpu()
        r = report(name, x, xr, dv)
        r["lat_std_raw"] = float(z.std())
        r["lat_std_scaled"] = r["lat_std_raw"] / float(cfg.vae_latent_scale)
        out["arms"].append(r)
        lat[name] = (x, xr, dv)

    cols = ["arm", "n", "in_c_norm", "in_trans_std", "divisor_mean_m", "lat_std_raw",
            "lat_std_scaled", "rec_trans", "rec_trans_m", "rec_rot6d", "rec_intr"]
    print("\n" + " ".join(f"{c:>22}" if c == "arm" else f"{c:>14}" for c in cols))
    for r in out["arms"]:
        print(" ".join(f"{r[c]:>22}" if c == "arm" else
                       (f"{r[c]:>14}" if isinstance(r[c], int) else f"{r[c]:14.6f}") for c in cols))

    # ---- ratio 버킷: swap arm 이 어느 배율부터 무너지는지 ----
    x, xr, dv = lat["swap(context align)"]
    d = (xr - x).abs()
    rt = d[..., 6:9].mean(dim=(1, 2)); rr = d[..., 0:6].mean(dim=(1, 2))
    edges = [0.0, 0.5, 0.8, 1.25, 2.0, 4.0, 1e9]
    print(f"\n{'ratio(own/ctx)':>18} {'n':>6} {'in_c_norm':>12} {'rec_trans':>12} "
          f"{'rec_trans_m':>12} {'rec_rot6d':>12}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ratio >= lo) & (ratio < hi)
        if m.sum() == 0:
            continue
        b = {"lo": lo, "hi": (None if hi > 1e8 else hi), "n": int(m.sum()),
             "in_c_norm": float(x[m][..., 6:9].norm(dim=-1).mean()),
             "rec_trans": float(rt[m].mean()),
             "rec_trans_m": float((rt[m] * dv[m]).mean()),
             "rec_rot6d": float(rr[m].mean())}
        out["ratio_buckets"].append(b)
        lab = f"[{lo:g}, {'inf' if hi > 1e8 else f'{hi:g}'})"
        print(f"{lab:>18} {b['n']:>6} {b['in_c_norm']:12.6f} {b['rec_trans']:12.6f} "
              f"{b['rec_trans_m']:12.6f} {b['rec_rot6d']:12.6f}")

    od = os.path.join(HERE, "results", "scene_decoupled")
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "vae_scale_swap.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
