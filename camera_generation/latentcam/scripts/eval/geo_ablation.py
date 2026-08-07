"""geo conditioning 이 **실제로 쓰이는지** 재는 ablation.

배경: da3_1k_customgeo_withs / _nos 는 context view 선택만 다른데 val/loss_traj 가
구분이 안 됐다(epoch 1~35 에서 withs 가 더 낮은 epoch 20/35, 차이 sd 0.0038). GeoTokenizer
가중치는 초기값 대비 확실히 움직였고 두 arm 이 서로 갈라졌으므로 "인코더가 안 배운다"는
아니다. 남은 가설은 "diffusion 모델이 geo 토큰을 별로 안 쓴다" 쪽이고 이 스크립트가 그걸 잰다.

재는 방법 — **paired** 비교다. 같은 배치 / 같은 noise / 같은 timestep 에서 geo 토큰만 바꾼다.
  real     그대로
  shuffle  배치 축으로 roll(1). 토큰 **통계는 그대로**이고 scene 짝만 틀린다. 이게 핵심
           조건이다: zero 는 분포까지 바꿔 버려서 "분포가 이상해서 나빠진 것"과 "내용을
           안 봐서 그런 것"을 구분 못 하는데, shuffle 은 내용 정합성만 깬다.
  zero     0 (geo_proj 에 bias 가 있으므로 '조건 없음'이 아니라 '상수 토큰'이다)
  none     geo_emb=None. cross-attn 블록 자체를 건너뛴다 — 학습 때 본 적 없는 경로라
           참고용이고, 위 셋끼리의 비교가 본론이다.

지표
  eps_mse            학습 목적함수 그대로 (MSE(noise_pred, noise)). timestep 을 고정 격자로
                     쓸어서 샘플링 노이즈를 없앴기 때문에 val/loss_traj 보다 훨씬 민감하다.
  d_pred             ||pred - pred_real|| / ||pred_real||. geo 를 바꿨을 때 **출력이 움직이는
                     양** 자체. 0 에 가까우면 모델이 geo 를 안 본다는 직접 증거다.
  resid_ratio        geo cross-attn 의 ||a|| / ||x|| (layer 별). CrossAttention 이
                     norm(x + a) 라서 이게 geo 가 hidden state 를 바꾸는 실제 크기다.
                     attention weight 는 softmax 라 항상 합이 1 이므로 "geo 를 얼마나
                     쓰는지"를 못 잰다 — 반드시 이 값으로 봐야 한다.
  attn_entropy       geo 토큰 M 개에 대한 attention 분포의 엔트로피 (nats). ln(M) 이면 완전
                     균일(= 아무 데도 안 보는 것과 같음), 낮으면 특정 patch 에 집중.
  view_mass          M = V x P 로 접어 view 별 attention 질량. withs arm 에서 view0 (= target
                     첫 카메라) 질량이 1/V 보다 유의하게 크면 그 view 를 실제로 특별 취급하는 것.

사용
  PYTHONPATH=..:.:../data python scripts/eval/geo_ablation.py \
      --run results/20260807_112857_da3_1k_customgeo_withs --ckpt last.pth --gpu 2 --batches 20
"""

import argparse
import json
import math
import os
import os.path as osp
import sys

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
MAIN = osp.join(REPO, 'main')
sys.path.insert(0, MAIN)
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(REPO, 'data'))
sys.path.insert(0, osp.join(REPO, 'scripts'))

CONDS = ('real', 'shuffle', 'zero', 'none')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='results/<run_dir> (config.yaml + ckpts/ 필요)')
    ap.add_argument('--ckpt', default='last.pth', help='<run>/ckpts 아래 파일')
    ap.add_argument('--gpu', default=None, help='CUDA_VISIBLE_DEVICES (torch import 전에 건다)')
    ap.add_argument('--batches', type=int, default=20, help='val 배치 수')
    ap.add_argument('--timesteps', default='50,150,250,350,450,550,650,750,850,950',
                    help='고정 timestep 격자 (배치마다 전부 쓴다)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0, help='noise 재현용 (조건 간 동일 noise)')
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    run_dir = osp.normpath(args.run if osp.isabs(args.run) else osp.join(REPO, args.run))
    ckpt_path = osp.join(run_dir, 'ckpts', args.ckpt)
    for p in (osp.join(run_dir, 'config.yaml'), ckpt_path):
        if not osp.isfile(p):
            raise SystemExit(f"missing: {p}")

    from eval_testset import build_cfg              # 저장된 config <- 현재 기본값 병합 로직 재사용
    cfg, cfg_dict = build_cfg(run_dir, {})
    os.chdir(MAIN)

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    import train_latent_cam_dm as T
    T.cfg = cfg
    from base import Trainer
    from models.vae_intr_large import CameraVAE
    from models.t5 import T5EncoderModel
    from models.camera_diffusion_model_latent import CameraDiffusionModel
    from models.geo_encoder import build_geo_encoder
    from diffusers import DDPMScheduler, DDIMScheduler

    torch.manual_seed(cfg.random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sched_kw = dict(num_train_timesteps=cfg.diffusion_max_step, prediction_type=cfg.prediction_type,
                    beta_schedule=cfg.beta_schedule, clip_sample=cfg.clip_sample,
                    set_alpha_to_one=cfg.set_alpha_to_one, steps_offset=cfg.steps_offset,
                    beta_start=cfg.beta_start, beta_end=cfg.beta_end)
    noise_scheduler = (DDPMScheduler(**sched_kw) if cfg.sampling_type == 'ddpm'
                       else DDIMScheduler(**sched_kw))
    if cfg.prediction_type != 'epsilon':
        raise SystemExit(f"prediction_type={cfg.prediction_type}: 이 스크립트는 eps 타깃만 잰다")

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator(include_train=False)
    valid_dataloader = trainer.valid_batch_generator

    geo_encoder = build_geo_encoder(cfg).to(device) if getattr(cfg, 'geo_encoder', None) else None
    if geo_encoder is None:
        raise SystemExit('geo_encoder 가 없는 run 이다 — 잴 게 없다')
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len, dtype=cfg.t5_dtype, device=device,
        checkpoint_path=osp.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
        tokenizer_path=osp.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path), shard_fn=None)

    _geo_kw = {}
    if getattr(cfg, 'geo_cam_embed', None):
        _raw = 6 if cfg.geo_cam_embed == 'plucker' else 11
        _geo_kw = dict(geo_latent_dim=getattr(cfg, 'geo_latent_dim', 768), geo_cam_raw_dim=_raw,
                       geo_cam_embed_dim=getattr(cfg, 'geo_cam_embed_dim', 128))
    model = CameraDiffusionModel(cam_dim=cfg.cam_dim, **_geo_kw)
    sd = torch.load(ckpt_path, map_location='cpu')
    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()

    if getattr(geo_encoder, 'trainable', False):
        geo_ckpt = ckpt_path.replace('.pth', '_geo.pth')
        if not osp.isfile(geo_ckpt):
            raise SystemExit(f"{geo_ckpt} 가 없다 — 랜덤 가중치로 잴 수 없다")
        _gm, _gu = geo_encoder.load_state_dict(torch.load(geo_ckpt, map_location='cpu'), strict=False)
        if _gu:
            raise SystemExit(f"{geo_ckpt}: unexpected keys {_gu[:5]}")
        print(f"(geo) loaded {geo_ckpt} (missing={len(_gm)} = frozen DINO)")
        geo_encoder = geo_encoder.to(device).eval()

    camera_vae = None
    if cfg.use_vae:
        camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
        camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
        camera_vae.eval()

    # ---- geo cross-attn 계측 훅 -------------------------------------------------
    # CrossAttention.forward 는 norm(x + a) 라서 a 를 밖에서 볼 수 없다. 내부의
    # nn.MultiheadAttention 에 훅을 걸면 입력 x(args[0])와 출력 (a, attn_weight)를 같이 잡는다.
    # 모델 코드를 안 건드리는 게 목적이다 (기존 run 재현성 유지).
    probe = {}

    def _mk_hook(li):
        def hook(mod, inputs, output):
            x = inputs[0]
            a, w = output[0], output[1]
            d = probe.setdefault(li, {})
            d['resid'] = (a.float().norm(dim=-1) / x.float().norm(dim=-1).clamp(min=1e-12)).mean().item()
            if w is not None:
                d['attn'] = w.detach().float()          # (B, T, M)
        return hook

    handles = [model.layers[li][5].attn.register_forward_hook(_mk_hook(li))
               for li in range(len(model.layers))]

    ts_grid = [int(x) for x in args.timesteps.split(',')]
    acc = {c: {'eps_mse': 0.0, 'd_pred': 0.0, 'n': 0} for c in CONDS}
    resid = {li: 0.0 for li in range(len(model.layers))}
    ent_sum = 0.0
    view_mass = None
    n_probe = 0
    V = P = None

    gen = torch.Generator(device='cpu').manual_seed(args.seed)

    with torch.no_grad():
        pbar = tqdm(valid_dataloader, total=args.batches)
        for step, data in enumerate(pbar):
            if step >= args.batches:
                break
            traj = data['cam_param'].to(device)
            B = traj.shape[0]
            if B < 2:
                continue                                # shuffle 이 성립 안 함
            text_embeds, text_masks = text_encoder(data['text_prompt'], device)
            text_embeds = text_embeds.float()
            text_masks = text_masks.bool()

            if 'geo_emb' in data:
                geo_emb, geo_mask = T.geo_emb_from_cache(data, device)
            else:
                geo_emb, geo_mask = T.geo_encode(geo_encoder, data, device)
            geo_emb = T.attach_geo_cam(geo_emb, data, device)

            z = camera_vae.encode(traj) / cfg.vae_latent_scale if cfg.use_vae else traj
            noise = torch.randn(z.shape, generator=gen).to(device)   # 조건 간 **동일** noise

            variants = {
                'real':    (geo_emb, geo_mask),
                'shuffle': (torch.roll(geo_emb, 1, 0), torch.roll(geo_mask, 1, 0)),
                'zero':    (torch.zeros_like(geo_emb), geo_mask),
                'none':    (None, None),
            }

            for t in ts_grid:
                ts = torch.full((B,), t, device=device, dtype=torch.long)
                x_t = noise_scheduler.add_noise(z, noise, ts)
                pred_real = None
                for c in CONDS:
                    ge, gm = variants[c]
                    probe.clear()
                    pred = model(x_t, ts.float(), text_embeds, text_masks, ge, gm)
                    acc[c]['eps_mse'] += F.mse_loss(pred, noise).item() * B
                    acc[c]['n'] += B
                    if c == 'real':
                        pred_real = pred
                        for li, d in probe.items():
                            resid[li] += d['resid']
                        w = probe[0].get('attn')         # layer 0 의 (B,T,M) — 대표로 본다
                        if w is not None:
                            p = w.mean(dim=1)            # (B,M) target token 평균
                            p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
                            ent_sum += (-(p * p.clamp(min=1e-12).log()).sum(-1)).mean().item()
                            if V is None:
                                V = int(data['geo_plucker_map'].shape[1]) if 'geo_plucker_map' in data else 6
                                P = p.shape[-1] // V
                                view_mass = torch.zeros(V)
                            view_mass += p.view(B, V, P).sum(-1).mean(0).cpu()
                            n_probe += 1
                    else:
                        acc[c]['d_pred'] += ((pred - pred_real).norm() /
                                             pred_real.norm().clamp(min=1e-12)).item() * B
            pbar.set_description(f"batch {step}")

    for h in handles:
        h.remove()

    n_layers = len(model.layers)
    res = {
        'run': run_dir, 'ckpt': args.ckpt, 'batches': args.batches, 'timesteps': ts_grid,
        'geo_encoder': cfg.geo_encoder,
        'geo_first_view_target_s': getattr(cfg, 'geo_first_view_target_s', None),
        'conds': {c: {'eps_mse': acc[c]['eps_mse'] / max(acc[c]['n'], 1),
                      'd_pred_vs_real': acc[c]['d_pred'] / max(acc[c]['n'], 1)}
                  for c in CONDS},
        'resid_ratio_per_layer': [resid[li] / max(n_probe, 1) for li in range(n_layers)],
        'attn_entropy_nats': ent_sum / max(n_probe, 1),
        'attn_entropy_uniform_nats': math.log(V * P) if V else None,
        'view_mass': (view_mass / max(n_probe, 1)).tolist() if view_mass is not None else None,
        'view_mass_uniform': 1.0 / V if V else None,
    }

    out_dir = args.out or osp.join(REPO, 'eval_my', 'geo_ablation',
                                   f"{osp.basename(run_dir)}__{args.ckpt[:-4]}")
    out_dir = out_dir if osp.isabs(out_dir) else osp.join(REPO, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, 'geo_ablation.json'), 'w') as f:
        json.dump(res, f, indent=2)

    print(f"\n===== {osp.basename(run_dir)} / {args.ckpt} "
          f"(batches={args.batches}, timesteps={len(ts_grid)}) =====")
    base = res['conds']['real']['eps_mse']
    print(f"{'cond':9s} {'eps_mse':>10s} {'vs real':>10s} {'d_pred':>10s}")
    for c in CONDS:
        d = res['conds'][c]
        print(f"{c:9s} {d['eps_mse']:10.6f} {d['eps_mse'] - base:+10.6f} {d['d_pred_vs_real']:10.6f}")
    print(f"\ngeo cross-attn ||a||/||x|| (layer 0..{n_layers - 1}):")
    print('  ' + ' '.join(f"{v:.4f}" for v in res['resid_ratio_per_layer']))
    print(f"\nattn entropy {res['attn_entropy_nats']:.4f} nats "
          f"(균일분포 = {res['attn_entropy_uniform_nats']:.4f})")
    if res['view_mass']:
        print("view mass  " + ' '.join(f"v{i}={m:.4f}" for i, m in enumerate(res['view_mass'])) +
              f"   (균일 = {res['view_mass_uniform']:.4f})")
    print(f"\n-> {osp.join(out_dir, 'geo_ablation.json')}")


if __name__ == '__main__':
    main()
