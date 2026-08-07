"""Full held-out-split inference + CLaTr/caption evaluation for a FINISHED run.

Why this exists: train_latent_cam_dm.run_validation only ever walks cfg.val_max_batches (=20)
x batch_size (=8) = 160 of the ~4k held-out segments, so every wandb val point is a 160-sample
estimate and epoch-to-epoch swings of ~1 sd swamp most arm-vs-arm gaps. This replays the SAME
validation path over the WHOLE held-out split, once, for a given checkpoint.

Config: taken from the RUN'S OWN saved results/<run>/config.yaml (train_latent_cam_dm.py:239
dumps the fully resolved config), layered on top of the CURRENT conf/config.yaml defaults so
keys added after the run still resolve. That matters here: the three finished geo runs were all
trained with vae_latent_scale 0.96032625, which conf/experiment/geo_worldtraj.yaml no longer
carries -- reading the live experiment yaml would silently evaluate with 0.47637.

Split: base.Trainer._make_batch_generator() is reused verbatim, so the random_split(seed=42,
train_frac) held-out set is bit-identical to the one the run validated on -- this is a superset
of the 160 segments wandb saw, not a different set.

The sampling / decode / out_to_trajectory / json-dump / CLaTr-subprocess chain is
train_latent_cam_dm's helpers imported directly (module-global `cfg` rebound), not a copy, so
the two cannot drift.

Usage (from anywhere; the script chdir's to main/ because the CLaTr subprocesses are launched
with cwd-relative paths):
  python scripts/eval_testset.py --run results/20260731_001511_dl3dv_geo_worldtraj
  python scripts/eval_testset.py --run <dir> --ckpt last.pth --gpu 3 --max-batches 3   # smoke

Output: eval_my/<run>__<ckpt>/ with config.yaml (what was actually used),
eval_meta.json, test/ (per-sample caption + ref/pred transforms = the inputs), seq/ token/
(CLaTr text feats), preds.npy, metrics.json, preds_scores.csv, losses.json.
"""
import argparse
import datetime
import json
import os
import os.path as osp
import sys
import time

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..'))
EVAL_ROOT = 'eval_my'     # 2026-08-02: was results/testset_eval; --out still overrides
MAIN = osp.join(REPO, 'main')
sys.path.insert(0, MAIN)
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, 'data'))

# Legacy config keys -> their current names. The value is carried over as-is; a run whose saved
# config predates a rename would otherwise fall back to the CURRENT default and change behaviour.
RENAMED = {'geo_anchor_first_frame': 'geo_cover_centered_at_s'}


def build_cfg(run_dir, overrides):
    """Current conf/config.yaml defaults <- run's saved config.yaml <- CLI overrides."""
    from omegaconf import OmegaConf
    from hydra_cfg import load_cfg, _finalize_dict
    import torch
    cfg, cfg_dict = load_cfg('config', overrides=[])          # base defaults only
    saved = OmegaConf.to_container(OmegaConf.load(osp.join(run_dir, 'config.yaml')), resolve=True)
    d = dict(cfg_dict)
    unknown = []
    for k, v in saved.items():
        k = RENAMED.get(k, k)
        if k not in d:
            unknown.append(k)
        d[k] = v
    if unknown:
        print(f"[cfg] keys in the saved config that no longer exist in conf/config.yaml: {unknown}")
    missing = [k for k in cfg_dict if k not in saved and RENAMED.get(k, k) not in saved]
    if missing:
        print(f"[cfg] keys added AFTER this run (kept at current defaults): {sorted(missing)}")
    d.update(overrides)
    _finalize_dict(d)
    ns = dict(d)
    ns['t5_dtype'] = getattr(torch, d['t5_dtype']) if isinstance(d['t5_dtype'], str) else d['t5_dtype']
    from types import SimpleNamespace
    return SimpleNamespace(**ns), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='results/<run_dir> (must contain config.yaml + ckpts/)')
    ap.add_argument('--ckpt', default='best.pth', help='file under <run>/ckpts (best.pth | last.pth)')
    ap.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES (set before torch import)')
    ap.add_argument('--out', default=None, help='output dir (default eval_my/<run>__<ckpt>)')
    ap.add_argument('--batch-size', type=int, default=None,
                    help='override eval batch size. NOTE: sample() draws randn(B,...) so a '
                         'different B = a different noise stream = different (equally valid) samples.')
    ap.add_argument('--max-batches', type=int, default=None, help='cap #batches (smoke test only)')
    ap.add_argument('--skip-clatr', action='store_true', help='dump jsons + losses, skip the metric subprocesses')
    # [new 2026-08-03] arbitrary config override, mainly for train_seg_list / test_seg_list.
    # The 07-30~07-31 runs saved seg_list null, so _make_batch_generator falls back to
    # random_split on TODAY's pool (39817) -- a different partition than the 39830 those runs
    # actually split, and 89.48% of it was their own TRAIN. Pinning
    # latentcam_{train,test}_seg_list_c4d2k5y4.txt restores their real held-out set.
    # Value is parsed as YAML so null / true / 3 / 0.5 keep their types.
    ap.add_argument('--set', dest='sets', action='append', default=[], metavar='KEY=VALUE',
                    help='override any config key (repeatable), e.g. --set test_seg_list=/path/x.txt')
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    run_dir = args.run if osp.isabs(args.run) else osp.join(REPO, args.run)
    run_dir = osp.normpath(run_dir)
    ckpt_path = osp.join(run_dir, 'ckpts', args.ckpt)
    for p in (osp.join(run_dir, 'config.yaml'), ckpt_path):
        if not osp.isfile(p):
            raise SystemExit(f"missing: {p}")

    ov = {}
    if args.batch_size:
        ov['batch_size'] = args.batch_size
    if args.sets:
        import yaml as _yaml
        for kv in args.sets:
            if '=' not in kv:
                raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
            k, v = kv.split('=', 1)
            ov[k.strip()] = _yaml.safe_load(v)
        print(f"[cfg] --set overrides: { {k: ov[k] for k in ov} }")
    cfg, cfg_dict = build_cfg(run_dir, ov)

    os.chdir(MAIN)      # CLaTr / eval subprocesses are launched with cwd-relative work dirs

    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from tqdm import tqdm

    # Reuse the trainer's own helpers so this eval cannot drift from run_validation.
    # Importing the module composes the BASE config into its module-global `cfg`; we rebind it
    # to the run's config below, before anything reads it.
    import train_latent_cam_dm as T
    T.cfg = cfg
    from base import Trainer
    from models.vae_intr_large import CameraVAE
    from models.t5 import T5EncoderModel
    from models.camera_diffusion_model_latent import CameraDiffusionModel
    from models.geo_encoder import build_geo_encoder
    from utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
    from utils.eval_utils import run_command_in_dir
    from diffusers import DDPMScheduler, DDIMScheduler

    torch.manual_seed(cfg.random_seed)

    tag = f"{osp.basename(run_dir)}__{args.ckpt[:-4]}"
    out_dir = args.out or osp.join(REPO, EVAL_ROOT, tag)
    # 상대경로 --out 은 REPO 기준. os.chdir(MAIN) 이 이미 돌았기 때문에 그냥 abspath 하면
    # main/ 밑으로 떨어진다 (기본값은 REPO 를 붙여서 만드니 영향 없고, --out 을 준 경우만 문제).
    out_dir = out_dir if osp.isabs(out_dir) else osp.join(REPO, out_dir)
    eval_data_dir = osp.join(out_dir, 'test')
    os.makedirs(eval_data_dir, exist_ok=True)
    from hydra_cfg import save_cfg_yaml
    save_cfg_yaml(cfg_dict, osp.join(out_dir, 'config.yaml'))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    traj_len = cfg.num_cam

    sched_kw = dict(num_train_timesteps=cfg.diffusion_max_step, prediction_type=cfg.prediction_type,
                    beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
                    set_alpha_to_one=cfg.set_alpha_to_one, steps_offset=cfg.steps_offset,
                    beta_start=cfg.beta_start, beta_end=cfg.beta_end)
    noise_scheduler = (DDPMScheduler(**sched_kw) if cfg.sampling_type == 'ddpm'
                       else DDIMScheduler(**sched_kw))

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator(include_train=False)      # same random_split as training
    valid_dataloader = trainer.valid_batch_generator
    n_val = len(trainer.validset_loader)
    print(f"[split] held-out segments: {n_val} (training validated on the first "
          f"{cfg.val_max_batches * cfg.batch_size})")

    geo_encoder = build_geo_encoder(cfg).to(device) if getattr(cfg, 'geo_encoder', None) else None
    if cfg.load_points:
        raise SystemExit('load_points runs are not supported here (no pc_encoder wiring)')
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
        checkpoint_path=osp.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=osp.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)

    _geo_kw = {}
    if getattr(cfg, 'geo_cam_embed', None):
        # relfirst = 11 per-view dims, plucker = 6 per-patch dims (train_latent_cam_dm.py 와 동일)
        _raw = 6 if cfg.geo_cam_embed == 'plucker' else 11
        _geo_kw = dict(geo_latent_dim=getattr(cfg, 'geo_latent_dim', 768), geo_cam_raw_dim=_raw,
                       geo_cam_embed_dim=getattr(cfg, 'geo_cam_embed_dim', 128))
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim, **_geo_kw)
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    model.load_state_dict(sd, strict=True)              # strict: a shape/name drift must not pass silently
    model = model.to(device).eval()

    # [new 2026-08-07] geo_encoder='custom' 은 학습되는 인코더라 가중치를 같이 복원해야 한다.
    # train_latent_cam_dm 이 <ckpt>.pth 옆에 <last|best>_geo.pth 로 남긴다 (frozen DINO 는 뺀
    # 상태라 missing 키가 나오는 게 정상 -> strict=False). 파일이 없으면 랜덤 초기값으로 평가하는
    # 셈이라 조용히 넘어가면 안 되고 여기서 죽인다.
    if getattr(geo_encoder, 'trainable', False):
        geo_ckpt = ckpt_path.replace('.pth', '_geo.pth')
        if not osp.isfile(geo_ckpt):
            raise SystemExit(f"geo_encoder='{cfg.geo_encoder}' 는 학습된 인코더인데 "
                             f"{geo_ckpt} 가 없다 — 랜덤 가중치로 평가할 수 없다")
        _gm, _gu = geo_encoder.load_state_dict(
            torch.load(geo_ckpt, map_location='cpu'), strict=False)
        if _gu:
            raise SystemExit(f"{geo_ckpt}: unexpected keys {_gu[:5]}")
        print(f"(geo) loaded {geo_ckpt} (missing={len(_gm)} = frozen DINO)")
        geo_encoder = geo_encoder.to(device).eval()

    camera_vae = None
    if cfg.use_vae:
        camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
        camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
        camera_vae.eval()

    clip_model = T.load_clip_model(cfg.clip_version, device=device) if T._HAS_CLIP else None
    if clip_model is None and not args.skip_clatr:
        raise SystemExit('clip_extraction unavailable -> CLaTr feats cannot be written')

    meta = {
        'run_dir': run_dir, 'ckpt': ckpt_path,
        'ckpt_mtime': datetime.datetime.fromtimestamp(osp.getmtime(ckpt_path)).isoformat(timespec='seconds'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'started': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_heldout_segments': n_val, 'batch_size': cfg.batch_size,
        'max_batches': args.max_batches, 'argv': sys.argv,
        'vae_latent_scale': cfg.vae_latent_scale, 'vae_ckpt_path': cfg.vae_ckpt_path,
        'scale_mode': cfg.scale_mode, 'intr_norm': cfg.intr_norm,
        'sampling_type': cfg.sampling_type, 'diffusion_inference_step': cfg.diffusion_inference_step,
    }
    _rp = osp.join(run_dir, 'ckpts', 'resume.pth')      # epoch/step/wandb_id the run reached
    if osp.isfile(_rp):
        _r = torch.load(_rp, map_location='cpu')
        meta['resume'] = {k: _r.get(k) for k in ('epoch', 'global_step', 'best_val', 'wandb_id', 'exp_name')}
    with open(osp.join(out_dir, 'eval_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    tot_lat = tot_traj = 0.0
    n_seen = 0
    names = []
    t0 = time.time()
    with torch.no_grad():
        for step, data in enumerate(tqdm(valid_dataloader, total=args.max_batches or len(valid_dataloader))):
            if args.max_batches is not None and step >= args.max_batches:
                break
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(device)
            E0 = data['first_extrinsic'].to(device)
            scale = data['avg_scale'].to(device)
            width = data['width'].to(device)
            height = data['height'].to(device)
            intrinsics = data['intrinsics'].to(device)
            B = traj.shape[0]
            # [new] geo_return_idxs side-channel (absent unless the flag is on). geo_swapped is
            # only present under geo_swap_mode; default 0 = "not swapped" so the dump is uniform.
            geo_ctx_c2w = data.get('geo_ctx_c2w')
            geo_idxs_np = data['geo_idxs'].tolist() if 'geo_idxs' in data else [None] * B
            geo_swapped_np = (data['geo_swapped'].tolist() if 'geo_swapped' in data
                              else [0] * B)

            pc_embeds, pc_masks = None, None
            if 'geo_emb' in data:
                pc_embeds, pc_masks = T.geo_emb_from_cache(data, device)
            elif geo_encoder is not None and 'images' in data:
                pc_embeds, pc_masks = T.geo_encode(geo_encoder, data, device)
            if pc_embeds is not None:
                pc_embeds = T.attach_geo_cam(pc_embeds, data, device)

            text_embeds, text_masks = text_encoder(text_prompt, device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()
            traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale if cfg.use_vae else traj

            if getattr(cfg, 'is_ar', False):
                out = T.ar_sample(model, noise_scheduler, traj_len, text_embeds, text_masks,
                                  pc_embeds, pc_masks, cfg.ar_chunk_size, cfg.cam_dim)
            else:
                out = T.sample(model, noise_scheduler, traj_len, text_embeds, text_masks,
                               pc_embeds, pc_masks)

            tot_lat += F.mse_loss(out, traj_latents, reduction='mean').item() * B
            traj_pred = camera_vae.decode(out * cfg.vae_latent_scale) if cfg.use_vae else out
            tot_traj += F.mse_loss(traj_pred, traj, reduction='mean').item() * B
            n_seen += B

            ref_intrinsics = make_intrinsics(traj[:, :, -2:], width, height, intrinsics).cpu().tolist()
            pred_intrinsics = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics).cpu().tolist()
            width = width.cpu().tolist()
            height = height.tolist()

            traj = out_to_trajectory(traj, scale, E0, device)
            traj_pred = out_to_trajectory(traj_pred, scale, E0, device)
            m_ref = inverse_camera_matrix(traj)
            m_pred = inverse_camera_matrix(traj_pred)
            m_ref[:, :, :3, 1:3] *= -1
            m_pred[:, :, :3, 1:3] *= -1
            m_ref = m_ref.cpu().tolist()
            m_pred = m_pred.cpu().tolist()

            if clip_model is not None:
                seq_embeds, tok_embeds = T.encode_text(text_prompt, clip_model, max_token_length=None,
                                                       device=device)
                T.save_feats_custom(seq_embeds, data_name, Path(osp.join(out_dir, 'seq')))
                T.save_feats_custom(tok_embeds, data_name, Path(osp.join(out_dir, 'token')))
            for i in range(len(data_name)):
                def _tj(mats, intr):
                    return {"w": width[i][0], "h": height[i][0],
                            "fl_x": intr[i][0][0][0], "fl_y": intr[i][0][1][1],
                            "cx": intr[i][0][0][2], "cy": intr[i][0][1][2],
                            "frames": [{"transform_matrix": mats[i][k], "monst3r_im_id": k + 1}
                                       for k in range(len(mats[i]))]}
                with open(osp.join(eval_data_dir, f"{data_name[i]}_caption.json"), 'w') as f:
                    json.dump({"Concise Interaction": text_prompt[i]}, f, indent=4)
                with open(osp.join(eval_data_dir, f"{data_name[i]}_transforms_ref.json"), 'w') as f:
                    json.dump(_tj(m_ref, ref_intrinsics), f, indent=4)
                with open(osp.join(eval_data_dir, f"{data_name[i]}_transforms_pred.json"), 'w') as f:
                    json.dump(_tj(m_pred, pred_intrinsics), f, indent=4)
                # [new] geo_return_idxs: the context views this prediction was conditioned on,
                # written next to the trajectories so an offline script can measure how close the
                # generated cameras sit to them. Same c2w OpenGL convention as the *_transforms_*
                # files above (m_ref/m_pred flip Y/Z at :272; geo_ctx_c2w is OpenCV, so flip here).
                if geo_ctx_c2w is not None:
                    g = geo_ctx_c2w[i].clone()
                    g[:, :3, 1:3] *= -1
                    os.makedirs(osp.join(out_dir, 'geo_ctx'), exist_ok=True)
                    with open(osp.join(out_dir, 'geo_ctx', f"{data_name[i]}.json"), 'w') as f:
                        json.dump({"geo_idxs": geo_idxs_np[i],
                                   "geo_swapped": geo_swapped_np[i],
                                   "norm_scale": float(scale[i]),
                                   "c2w": g.tolist()}, f)
            names += data_name

    losses = {'val/loss_latent': tot_lat / n_seen, 'val/loss_traj': tot_traj / n_seen,
              'n_samples': n_seen, 'sampling_sec': round(time.time() - t0, 1)}
    with open(osp.join(out_dir, 'losses.json'), 'w') as f:
        json.dump(losses, f, indent=2)
    print(f"[losses] {losses}")

    with open(osp.join(out_dir, 'test_valid.txt'), 'w', encoding='utf-8') as f:
        for nm in sorted(set(names)):
            f.write(nm + '\n')

    if args.skip_clatr:
        print(f"[done] --skip-clatr; jsons under {out_dir}")
        return

    for cmd, wd in ((["-m", "src.extraction", f"checkpoint_path={cfg.clatr_ckpt_path}",
                      f"data_dir={out_dir}"], 'evaluate/CLaTr'),
                    (["-m", "src.eval_only", "--pred_path", osp.join(out_dir, 'preds.npy')],
                     'evaluate/eval')):
        print(f"[run] (cwd={wd}) {' '.join(cmd)}", flush=True)
        so, se, rc = run_command_in_dir([sys.executable] + cmd, wd)
        print(so)
        if rc != 0:
            print(se)
            raise SystemExit(f"{cmd[1]} failed (rc={rc})")

    mp = osp.join(out_dir, 'metrics.json')
    if osp.isfile(mp):
        with open(mp) as f:
            metrics = json.load(f)
        metrics.update(losses)
        with open(osp.join(out_dir, 'metrics_full.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
    print(f"[done] {out_dir}")


if __name__ == '__main__':
    main()
