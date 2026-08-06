# Known issues (latentcam training)

## 🔴 (a) train/val split is per-SEGMENT → scene-level validation leakage  [PARKED]
`main/base.py:_make_batch_generator` does `random_split(CamDataset, [0.9, 0.1])`, but
`CamDataset` is indexed per prompt-SEGMENT (40,059 segments from 6,132 scenes ≈ 6.5
segments/scene). Random per-segment split puts different segments of the SAME scene in
both train and val → validation is essentially all "seen scenes".

Impact: `val/loss_*` and CLaTr metrics are optimistic; they do NOT measure novel-scene
generalization. Affects both geo and text-only runs.

Fix (when addressed): split by SCENE (group segments by `samples[i][0]` scene_idx, hold
out ~10% of scenes for val). Requires a `base.py` change + restart.
Status: recorded only per user (2026-07-18); not fixed, current runs left as-is.

## ✅ (b) trainer._make_model() allocated an unused model  [FIXED]
`train_latent_cam_dm.py` called `trainer._make_model()` (base.py → `DataParallel(get_model()).cuda()`,
get_model defaults to cam_dim=11), but training builds its own `model = CameraDiffusionModel(cam_dim=cfg.cam_dim)`.
The trainer model was never used → wasted GPU memory. Removed the call (2026-07-18).
Takes effect on next launch (currently-running jobs unaffected).

## 🟡 (c) VAE intrinsics not reconstructed
See `docs/vae_verification.md`. VAE outputs fx/w≈1.0 regardless of input (DL3DV≈0.45).
Mitigated by DL3DV being ~fixed-FoV; pose (rot/trans) unaffected. Same VAE as config_large.

## Notes: VAE / CLaTr vs config_large (verified 2026-07-18)
- VAE: config.py uses `vae_20260302_300.pth` (scale 0.96032625) = config_large's ACTIVE
  vae (`results_vae/20260302/300.pth`, same scale). Match. config_large also has many
  commented-out alternatives incl. DL3DV-specific VAEs (`20260316_073439_dl3dv_7K`, etc.)
  on ckd248/data2 paths (not verified present locally).
- CLaTr: config.py uses `clatr_epoch109_dl3dv_seg_2.ckpt` (DL3DV-specific); config_large
  uses `epoch139_large.ckpt` (= local `checkpoints/clatr_epoch139_large.ckpt`). Different.
  See verification below / chat.
