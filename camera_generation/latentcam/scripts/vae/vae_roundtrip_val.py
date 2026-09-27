"""FROZEN camera VAE 의 encode->decode 왕복만으로 val 궤적이 얼마나 망가지는가.

WHY: TRUMANS-Lite arm 에서 vae_latent_scale 을 실측값으로 고쳤더니 val/loss_traj 와 fcd 는
크게 떨어졌는데 captions/fscore 는 안 움직였다. 그러면 남은 용의자는 두 개다 —
(1) diffusion 이 아직 못 맞춘다, (2) **frozen VAE 자체가 caption 이 보는 양을 못 나른다**.
이 스크립트가 (2) 를 단독으로 잰다: diffusion 을 완전히 건너뛰고 GT cam_param 을
encode->decode 만 시킨 뒤, 학습 val 루프와 **똑같은** 후처리(out_to_trajectory ->
inverse_camera_matrix -> OpenGL 부호 뒤집기)를 거쳐 GT 와 비교한다. 즉 "완벽한 diffusion"
상한선이다. 여기서 이미 caption 이 안 나오면 diffusion 을 아무리 돌려도 안 나온다.

같이 재는 것:
  frame0 오차   — 예측 궤적의 첫 프레임이 GT 첫 프레임과 왜 어긋나는지. out_to_trajectory 는
                  E0 를 **곱하기만** 하고 frame0=identity 를 강제하지 않는다. 그래서 frame0
                  identity 는 모델이 재현해내야 하는 값이고, VAE 왕복만으로도 이미 어긋난다.
  tortuosity    — path_len / net_displacement. jitter 와 "너무 큼"을 가른다.
  5*dt_local    — caption.py:406 이 실제로 보는 양 (fps=5). static 임계 0.02 대비.

env: EXP (기본 trumans_lite_ctxuniform_vls019), SPLIT (test), DEVICE (cuda)
out -> results/compare/vae_roundtrip_<EXP>/{summary.csv, stdout 표}

사용 예:
  CUDA_VISIBLE_DEVICES=1 EXP=trumans_lite_ctxuniform_vls019 \
    python scripts/vae/vae_roundtrip_val.py
"""
import os, sys, json
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))
from hydra_cfg import load_cfg

EXP = os.environ.get("EXP", "trumans_lite_ctxuniform_vls019")
SPLIT = os.environ.get("SPLIT", "test")
DEVICE = os.environ.get("DEVICE", "cuda")
FPS = 5                 # caption.py:406
STATIC_TH = 0.02        # caption.py cam_static_threshold

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
# geo encoder 는 이 진단에 필요 없다 — 이미지 로딩을 끄면 dataset 이 훨씬 빨리 뜬다.
cfg.geo_encoder = None
cfg.load_points = False
# [2026-09-27, R65] cam_param 만 쓰므로 video/text/geo 캐시 로드를 끈다 (d200 코퍼스는 molmo2 캐시 305 GB).
# 출력에는 영향이 없다 — 이 스크립트는 cam_param/first_extrinsic/avg_scale 만 읽는다.
for _k in ("peav_video_cache_dir", "peav_text_cache", "geo_raw_cache_dir", "geo_latent_cache_dir",
           "text_emb_cache_path"):
    if hasattr(cfg, _k):
        setattr(cfg, _k, None)

from dataset_dl3dv import CamDataset
from models.vae_intr_large import CameraVAE
from utils.data_utils import out_to_trajectory, inverse_camera_matrix

OUT = os.path.join(HERE, "results", "compare", f"vae_roundtrip_{EXP}")
os.makedirs(OUT, exist_ok=True)

vae = CameraVAE(latent_dim=cfg.cam_dim).to(DEVICE)
vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=DEVICE))
vae.eval()
print(f"VAE  {cfg.vae_ckpt_path}\n     latent_dim={cfg.cam_dim}  vae_latent_scale={cfg.vae_latent_scale}")

ds = CamDataset(cfg, SPLIT)
print(f"dataset {SPLIT}: {len(ds)} samples\n")


def to_gl_c2w(traj9, scale, E0):
    """학습 val 루프(train_latent_cam_dm.py:727-733) 와 **같은** 후처리 -> OpenGL c2w (B,N,4,4)."""
    m = out_to_trajectory(traj9, scale, E0, DEVICE)
    m = inverse_camera_matrix(m)
    m[:, :, :3, 1:3] *= -1
    return m.detach().cpu().numpy().astype(np.float64)


def local_dt(gl):
    """caption metric 이 보는 per-frame local translation 5*dt (N-1,3)."""
    cv = gl.copy(); cv[:, :3, 1:3] *= -1
    w2c = np.linalg.inv(cv)
    return FPS * (np.linalg.inv(w2c[:-1]) @ w2c[1:])[:, :3, 3]


def fwd_flips(gl, scale):
    """[2026-09-27, R65] 첫 카메라 전방축(OpenGL -z) 위 순변위 / scale 과 그 축의 부호 변화 횟수.
    goal 4 (가까운 dolly 떨림) 진단 — diffusion 추론 쪽 분석(tmp/r63)과 같은 정의다."""
    c = gl[:, :3, 3]; f = -gl[0, :3, 2]; s = (c - c[0]) @ f; ds = np.diff(s)
    return float(s[-1] / scale), int(np.sum(np.diff(np.sign(ds[np.abs(ds) > 1e-6])) != 0))


def stats(gl):
    c = gl[:, :3, 3]
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    plen = float(seg.sum()); net = float(np.linalg.norm(c[-1] - c[0]))
    return plen, net, (plen / net if net > 1e-9 else np.inf)


rows = []
with torch.no_grad():
    for i in range(len(ds)):
        d = ds[i]
        traj = d['cam_param'][None].to(DEVICE).float()
        E0 = d['first_extrinsic'][None].to(DEVICE).float()
        sc = d['avg_scale'][None].to(DEVICE).float()

        lat = vae.encode(traj) / cfg.vae_latent_scale          # 학습이 diffusion 타깃으로 쓰는 것
        rt = vae.decode(lat * cfg.vae_latent_scale)            # "완벽한 diffusion" 의 출력

        g_ref = to_gl_c2w(traj, sc, E0)[0]
        g_rt = to_gl_c2w(rt, sc, E0)[0]

        dt0 = float(np.linalg.norm(g_rt[0, :3, 3] - g_ref[0, :3, 3]))
        R = g_ref[0, :3, :3].T @ g_rt[0, :3, :3]
        rot0 = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        rpl, rnet, rtor = stats(g_ref)
        ppl, pnet, ptor = stats(g_rt)
        rdt, pdt = local_dt(g_ref), local_dt(g_rt)
        lat_std = float(lat.std())
        _s = float(d['avg_scale'])
        gfwd, gflip = fwd_flips(g_ref, _s)
        pfwd, pflip = fwd_flips(g_rt, _s)
        rows.append(dict(
            name=d['data_name'], lat_std=lat_std, dt0=dt0, rot0_deg=rot0,
            gt_plen=rpl, rt_plen=ppl, plen_ratio=ppl / max(rpl, 1e-9),
            gt_tor=rtor, rt_tor=ptor,
            gt_dt_med=float(np.median(np.abs(rdt))), rt_dt_med=float(np.median(np.abs(pdt))),
            gt_static_frac=float((np.abs(rdt) < STATIC_TH).mean()),
            rt_static_frac=float((np.abs(pdt) < STATIC_TH).mean()),
            dt_l1=float(np.abs(rdt - pdt).mean()),
            gt_fwd=gfwd, rt_fwd=pfwd, gt_flips=gflip, rt_flips=pflip,
        ))

hdr = ["name", "lat_std", "dt0", "rot0_deg", "gt_plen", "plen_ratio", "gt_tor", "rt_tor",
       "gt_dt_med", "rt_dt_med", "gt_static_frac", "rt_static_frac", "dt_l1",
       "gt_fwd", "rt_fwd", "gt_flips", "rt_flips"]
print(f"{'target':>8s} " + " ".join(f"{h:>10s}" for h in hdr[1:]))
for r in rows:
    print(f"{r['name'][-8:]:>8s} " + " ".join(f"{r[h]:10.4f}" for h in hdr[1:]))
med = {h: float(np.median([r[h] for r in rows])) for h in hdr[1:]}
print(f"{'MEDIAN':>8s} " + " ".join(f"{med[h]:10.4f}" for h in hdr[1:]))

with open(os.path.join(OUT, "summary.csv"), "w") as f:
    f.write(",".join(hdr) + "\n")
    for r in rows:
        f.write(",".join(str(r[h]) for h in hdr) + "\n")
print(f"\n-> {OUT}/summary.csv")
