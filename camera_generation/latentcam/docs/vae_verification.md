# Camera VAE verification (vae_20260302_300.pth, cam_dim=64)

Verified with `scripts/verify_vae.py` on 38 real DL3DV-960 segments (current pipeline
cam_param). Reproduce: `PYTHONPATH="main:." python scripts/verify_vae.py`.

## Findings

### Reconstruction (per component)
| component | metric | verdict |
|---|---|---|
| rotation (rot6d) | MSE 2.0e-4, **geodesic mean 1.33°** (median 1.0°, p95 3.6°) | ✅ good |
| translation      | norm MSE 1.8e-3, mean err 0.035 (normalized units) | ✅ good |
| **intrinsics**   | MSE **0.166**, corr(GT, recon) **≈ -0.04** | ❌ **broken** |
| latent scale     | scaled std **1.015** (target ~1.0) — vae_latent_scale=0.96032625 | ✅ calibrated |

### ❌ Intrinsics not reconstructed
- GT `fx/w, fy/h = 0.45, 0.80` (DL3DV wide FoV) → VAE recon **0.99, 0.99 (constant)**,
  ignoring the input (correlation ≈ 0).
- Cause: this VAE was trained on a `fx/w ≈ 1.0` (~53° FoV) distribution; DL3DV is wide-FoV.
  Only 64-dim VAE available is this one (the other ckpt is 32-dim → cam_dim mismatch); no
  DL3DV-fitted VAE exists.
- **Mitigating fact**: DL3DV intrinsics are near-CONSTANT across scenes (per-sample std
  0.0076 / 0.0138) → essentially a fixed-FoV dataset. So the intrinsics channel carries
  almost no information; the diffusion model just learns a constant, and rot/trans (what
  matters) are unaffected. Generated intrinsics will be wrong (~1.0) but are overridable
  with the known DL3DV constant.

### Latent interpolation — ✅ OK
Decoded interpolations of two encoded segments morph smoothly into valid trajectories
(rot6d → Gram-Schmidt guarantees valid rotations). See `scripts/vae_verify.png`.

### Scene scale — ✅ applied (camera-based)
- dataset divides translation by camera-based `avg_scale` (mean 2.36, range 0.44–4.32) →
  VAE-input `|trans|` mean ≈ 1.0 (normalized). `out_to_trajectory` multiplies trans by
  `avg_scale` at decode (re-applied). Good translation recon confirms the camera-based
  scale is compatible with the VAE (scale itself is fine; only intrinsics are off).

## Status / TODO (parked per user, 2026-07-18)
Not acting now. When intrinsics fidelity is needed, either:
- **(quick)** override the intrinsics channel with the DL3DV constant (~0.45, 0.80) at
  inference/validation instead of trusting the VAE output; keep VAE for rot/trans; OR
- **(clean)** fine-tune / retrain the VAE on DL3DV cam_param (incl. DL3DV intrinsics).

Current training is unaffected for pose (rot/trans) learning; leave as-is.
