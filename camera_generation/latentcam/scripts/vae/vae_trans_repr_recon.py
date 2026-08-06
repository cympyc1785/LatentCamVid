"""Step 1: cam_param[6:9] 를 w2c translation t 로 둘 때와 카메라 중심 c 로 둘 때, 어느 쪽이
CameraVAE 로 재구성하기 쉬운가를 held-out segment 에서 직접 잰다.

Step 0(scripts/data/cam_repr_w2c_vs_c2w.py)은 학습 없이 데이터의 곡률만 봤고, 그건 학습 난이도의
약한 대리 지표였다. 여기서는 대리 지표를 버리고 두 규약으로 각각 fit 한 VAE 두 개의 실제 recon
오차를 비교한다.

용어
  t        w2c translation. t = -R c. 지금 프로덕션 규약 (trans_repr 'w2c').
  c        카메라 중심. c = -R^T t. 대안 규약 (trans_repr 'c2w').
           ||t|| == ||c|| 이므로 두 arm 의 타깃 크기 분포는 같다. 방향만 다르다.
  arm      (trans_repr, ckpt) 짝. A = w2c, B = c2w.

**대칭 평가**가 핵심이다. 각 arm 의 "자기 채널" L1 만 보면 규약마다 채널 의미가 달라 비교가
성립하지 않는다. 그래서 두 arm 모두에 대해 디코드 결과를 **두 표현 모두로 환산**해서 잰다:
  err_center   mean ||c_hat - c||   (정규화 단위)  <- 주 지표. 렌더/Plücker 가 쓰는 값이 c 라서.
  err_t        mean ||t_hat - t||   (정규화 단위)     대칭 확인용. 어느 규약도 유리한 지표를
                                                     고르지 않았다는 걸 보이기 위해 같이 낸다.
  err_rot_deg  R_hat 와 R 사이 geodesic 각 (deg)      translation 표현과 무관해야 정상.
  err_intr     intr 채널 2개의 L1
환산 방법: 디코드된 6d 로 R_hat 을 복원한 뒤
  arm A: t_hat = out[6:9],        c_hat = -R_hat^T t_hat
  arm B: c_hat = out[6:9],        t_hat = -R_hat  c_hat
즉 A 의 center 오차에는 R_hat 오차가 섞여 들어가고, B 의 t 오차에도 똑같이 섞인다. 이 비대칭이
바로 규약 선택의 실제 비용이고, 양쪽을 다 내야 공정하다.

GT 는 한 번만 만든다: 데이터셋을 trans_repr='w2c' 로 한 번 로드해 (R, t) 를 얻고 c 는 여기서
계산한다. 두 arm 이 문자 그대로 같은 segment, 같은 순서, 같은 norm_scale 을 본다.

env
  W2C_CKPT   arm A 체크포인트 (기본 my_checkpoints/vae_transrepr_w2c/last.pth)
  C2W_CKPT   arm B 체크포인트 (기본 my_checkpoints/vae_transrepr_c2w/last.pth)
  EXP        cfg 를 조립할 experiment 이름 (기본 vae_transrepr_w2c; test_seg_list 를 여기서 가져온다)
  N          평가할 최대 segment 수 (0 = test_seg_list 전부, 기본 0)
  BS         배치 크기 (기본 64)
out -> results/compare/cam_repr_w2c_vs_c2w/vae_step1.json (+ 표 출력)
"""
import os, sys, json
import numpy as np
import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "main"))
os.chdir(os.path.join(HERE, "main"))

from hydra_cfg import load_cfg

EXP = os.environ.get("EXP", "vae_transrepr_w2c")
N = int(os.environ.get("N", "0"))
BS = int(os.environ.get("BS", "64"))
CKPT = {
    "w2c": os.environ.get("W2C_CKPT", os.path.join(HERE, "my_checkpoints/vae_transrepr_w2c/last.pth")),
    "c2w": os.environ.get("C2W_CKPT", os.path.join(HERE, "my_checkpoints/vae_transrepr_c2w/last.pth")),
}
OUT = os.path.join(HERE, "results", "compare", "cam_repr_w2c_vs_c2w")
os.makedirs(OUT, exist_ok=True)

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
cfg.trans_repr = "w2c"            # GT 는 항상 w2c 로 받아서 c 는 여기서 유도한다
cfg.batch_size = BS

from base import Trainer
from models.vae_intr_large import CameraVAE
from utils.rotation_utils import compute_rotation_matrix_from_ortho6d

device = "cuda"


def load_vae(path):
    m = CameraVAE(latent_dim=cfg.cam_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def rot_from_6d(x6):
    """(B,T,6) -> (B,T,3,3). 6d 는 회전행렬의 첫 두 '열' (dataset_dl3dv 가 [:, :3, 0], [:, :3, 1])."""
    B, T, _ = x6.shape
    return compute_rotation_matrix_from_ortho6d(x6.reshape(B * T, 6)).reshape(B, T, 3, 3)


def geodesic_deg(Ra, Rb):
    tr = (Ra * Rb).sum(dim=(-2, -1))                       # trace(Ra^T Rb)
    return torch.rad2deg(torch.arccos(((tr - 1.0) * 0.5).clamp(-1.0, 1.0)))


def main():
    missing = [k for k, p in CKPT.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"ckpt 없음: {[CKPT[k] for k in missing]}")
    vae = {k: load_vae(p) for k, p in CKPT.items()}
    for k, p in CKPT.items():
        print(f"arm {k:3s}  {p}")

    trainer = Trainer(cfg)
    trainer._make_batch_generator(include_train=False, include_val=True)
    loader = trainer.valid_batch_generator

    acc = {k: dict(ec=[], et=[], er=[], ei=[], lat=[]) for k in CKPT}
    gt = dict(ct=[], tt=[])
    done = 0
    for data in loader:
        if N and done >= N:
            break
        x = data['cam_param'].to(device).float()            # (B,T,11) — trans 채널은 w2c t
        R = rot_from_6d(x[..., :6])                         # (B,T,3,3)
        t = x[..., 6:9]
        c = -torch.einsum('btji,btj->bti', R, t)            # -R^T t
        intr = x[..., 9:11]
        inp = {"w2c": torch.cat([x[..., :6], t, intr], -1),
               "c2w": torch.cat([x[..., :6], c, intr], -1)}

        for k, m in vae.items():
            with torch.no_grad():
                z = m.encode(inp[k])
                z = z.sample() if hasattr(z, "sample") else z
                rec = m.decode(z)
            Rh = rot_from_6d(rec[..., :6])
            if k == "w2c":
                th = rec[..., 6:9]
                ch = -torch.einsum('btji,btj->bti', Rh, th)
            else:
                ch = rec[..., 6:9]
                th = -torch.einsum('btij,btj->bti', Rh, ch)
            acc[k]['ec'] += (ch - c).norm(dim=-1).mean(dim=1).cpu().tolist()
            acc[k]['et'] += (th - t).norm(dim=-1).mean(dim=1).cpu().tolist()
            acc[k]['er'] += geodesic_deg(Rh, R).mean(dim=1).cpu().tolist()
            acc[k]['ei'] += (rec[..., 9:11] - intr).abs().mean(dim=(1, 2)).cpu().tolist()
            acc[k]['lat'].append(float(z.std()))
        gt['ct'] += c.norm(dim=-1).mean(dim=1).cpu().tolist()
        gt['tt'] += t.norm(dim=-1).mean(dim=1).cpu().tolist()
        done += x.shape[0]
        if done % (BS * 10) == 0:
            print(f"  {done}", flush=True)

    print(f"\n{done} held-out segments (exp={EXP}, test_seg_list)")
    print(f"GT scale: mean||c|| = {np.mean(gt['ct']):.5f}   mean||t|| = {np.mean(gt['tt']):.5f}"
          f"   (같아야 정상: ||t|| == ||c||)")
    hdr = (f"{'arm':5s} {'err_center':>11s} {'err_t':>10s} {'err_rot_deg':>12s} "
           f"{'err_intr':>9s} {'lat_std':>8s}")
    print(hdr); print("-" * len(hdr))
    res = {}
    for k in ("w2c", "c2w"):
        a = acc[k]
        res[k] = dict(ckpt=CKPT[k], n=len(a['ec']),
                      err_center=float(np.mean(a['ec'])), err_center_p50=float(np.median(a['ec'])),
                      err_t=float(np.mean(a['et'])), err_t_p50=float(np.median(a['et'])),
                      err_rot_deg=float(np.mean(a['er'])), err_intr=float(np.mean(a['ei'])),
                      lat_std=float(np.mean(a['lat'])))
        v = res[k]
        print(f"{k:5s} {v['err_center']:11.5f} {v['err_t']:10.5f} {v['err_rot_deg']:12.4f} "
              f"{v['err_intr']:9.5f} {v['lat_std']:8.4f}")
    rc = res['w2c']['err_center'] / max(res['c2w']['err_center'], 1e-12)
    rt = res['w2c']['err_t'] / max(res['c2w']['err_t'], 1e-12)
    print(f"\nw2c / c2w  err_center {rc:.3f}x   err_t {rt:.3f}x   (>1 이면 c2w 가 유리)")

    json.dump(dict(exp=EXP, n=done, cam_dim=cfg.cam_dim, scale_mode=cfg.scale_mode,
                   intr_norm=cfg.intr_norm, gt_mean_norm_c=float(np.mean(gt['ct'])),
                   gt_mean_norm_t=float(np.mean(gt['tt'])), arms=res,
                   ratio_err_center=rc, ratio_err_t=rt),
              open(os.path.join(OUT, "vae_step1.json"), "w"), indent=2)
    print("saved", os.path.join(OUT, "vae_step1.json"))


if __name__ == "__main__":
    main()
