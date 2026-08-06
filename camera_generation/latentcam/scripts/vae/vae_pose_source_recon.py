"""FROZEN 카메라 VAE 가 pose_source 별 cam_param 분포에서 아직 쓸 만한가?

cfg.vae_ckpt_path 는 얼려 둔 것이고 최상위 transforms.json + avg_scale 분포(정규화 ||c||~0.18)
에서 학습됐다. da3 arm 은 da3/avg_scale 을 분모로 쓰는데 그게 p50 3.2배 작아서 정규화 ||c|| 가
~0.46 으로 올라간다. VAE 를 다시 학습하지 않기로 했으므로, 실제로 무엇이 얼마나 나빠지는지와
**DM 쪽 상수 하나(vae_latent_scale)로 보정 가능한지**를 여기서 잰다.

보고 항목 (arm 마다):
  in_trans_std     VAE 입력의 translation 3채널 std
  in_c_norm        정규화 카메라 중심 크기 mean ||c||
  lat_std_raw      encode 직후 latent std (vae_latent_scale 로 나누기 전)
  lat_std_scaled   / cfg.vae_latent_scale 후 = **diffusion 타깃의 std**. 1.0 에서 멀면 조건수 나쁨
  vae_scale_for_1  lat_std_scaled 를 정확히 1.0 으로 만드는 vae_latent_scale 값
  rec_trans        trans 3채널 recon MAE (정규화 단위)
  rec_trans_world  위 * 그 arm 의 divisor 평균 -> arm 간 비교 가능한 단위
  rec_rot6d        회전 6채널 recon MAE (분모와 무관해야 정상)
  rec_intr         intr 2채널 recon MAE

env: EXPS (콤마구분, 기본 'da3_1k_textonly,da3_1k_da3pose'), N (기본 800), SPLIT (train), SEED (0)
out -> results/compare/camera_jumps/vae_pose_source_recon.json
"""
import os, sys, json, random
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

EXPS = os.environ.get("EXPS", "da3_1k_textonly,da3_1k_da3pose").split(",")
N = int(os.environ.get("N", "800"))
SPLIT = os.environ.get("SPLIT", "train")
SEED = int(os.environ.get("SEED", "0"))
OUT = os.path.join(HERE, "results", "compare", "camera_jumps")
os.makedirs(OUT, exist_ok=True)

from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE

device = "cuda"
res = {}
for exp in EXPS:
    exp = exp.strip()
    cfg, _ = load_cfg("config", overrides=[f"experiment={exp}"])
    vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
    vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
    vae.eval()

    ds = CamDataset(cfg, type=SPLIT)
    random.seed(SEED)
    idxs = random.sample(range(len(ds)), min(N, len(ds)))
    cams, scales = [], []
    for i in idxs:
        b = ds[i]
        cams.append(b["cam_param"])
        s = b.get("norm_scale", None)
        scales.append(float(s) if s is not None else float("nan"))
    x = torch.stack(cams).to(device)                       # (N,T,11)

    with torch.no_grad():
        z = vae.encode(x)
        if not torch.is_tensor(z):                          # (mu, logvar) 형태 대비
            z = z[0]
        xr = vae.decode(z)
    lat_raw = float(z.std())
    lat_scaled = lat_raw / float(cfg.vae_latent_scale)
    d = (xr - x).abs()
    div = float(np.nanmean(scales))
    rec_trans = float(d[..., 6:9].mean())
    res[exp] = {
        "pose_source": str(cfg.pose_source),
        "n": len(idxs),
        "in_trans_std": float(x[..., 6:9].std()),
        "in_c_norm": float(x[..., 6:9].norm(dim=-1).mean()),
        "divisor_mean": div,
        "lat_std_raw": lat_raw,
        "lat_std_scaled": lat_scaled,
        "vae_scale_for_1": lat_raw,
        "rec_trans": rec_trans,
        "rec_trans_world": rec_trans * div,
        "rec_rot6d": float(d[..., 0:6].mean()),
        "rec_intr": float(d[..., 9:11].mean()),
    }
    del vae
    torch.cuda.empty_cache()

cols = ["pose_source", "n", "in_trans_std", "in_c_norm", "divisor_mean", "lat_std_raw",
        "lat_std_scaled", "vae_scale_for_1", "rec_trans", "rec_trans_world", "rec_rot6d",
        "rec_intr"]
print(f"\n{'exp':>20} " + " ".join(f"{c:>15}" for c in cols))
for exp, r in res.items():
    print(f"{exp:>20} " + " ".join(
        (f"{r[c]:>15}" if isinstance(r[c], (str, int)) else f"{r[c]:15.5f}") for c in cols))
p = os.path.join(OUT, "vae_pose_source_recon.json")
json.dump(res, open(p, "w"), indent=2)
print(f"\nsaved {p}")
