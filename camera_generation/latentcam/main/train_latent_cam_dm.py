import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import wandb
import datetime
import time
import json
import random
from pathlib import Path
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import gather_object, tqdm, broadcast_object_list
from torch_scatter import scatter_mean
import torch.nn as nn

import importlib as _importlib
from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config')
from base import Trainer  # ported to main/base.py (DL3DV dataset)
from models.vae_intr_large import CameraVAE
from models.t5 import T5EncoderModel
from utils.data_utils import out_to_trajectory, make_intrinsics, inverse_camera_matrix
from utils.eval_utils import run_command_in_dir
from utils.pc_utils import get_ray_sim_per_point
from diffusers import DDPMScheduler, DDIMScheduler

# CLaTr feature extraction (OpenAI clip) is optional — only needed for CLaTr eval.
try:
    from evaluate.CLaTr.clip_extraction import load_clip_model, encode_text, save_feats_custom
    _HAS_CLIP = True
except Exception as _e:
    print(f"(train) clip_extraction unavailable ({_e}); CLaTr feature saving disabled")
    _HAS_CLIP = False

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)
random.seed(cfg.random_seed)

# Camera diffusion denoiser: use the latent (geo-conditioned) variant.
from models.camera_diffusion_model_latent import CameraDiffusionModel

# Geo encoder (image-based scene encoder feeding the latent model's geo latent).
from models.geo_encoder import build_geo_encoder

# Legacy point-cloud encoder — only imported when the pc path is used (load_points).
# The geo path (lagernvs) does not need concerto, so we skip the import otherwise.
PCEncoder = None
if cfg.load_points:
    if cfg.point_encoder == 'concerto':
        from models.pc_encoder import PCEncoder
    elif cfg.point_encoder == 'custom':
        from models.pc_encoder_custom import PCEncoder
    elif cfg.point_encoder == 'mosaic':
        from models.pc_encoder_mosaic import PCEncoder

def geo_emb_from_cache(data, device):
    """Precomputed geo path: the dataset already read the frozen geo_emb off disk, so there is no
    encoder forward. Mirrors GeoEncoder.forward's contract -> (B, M, 768), (B, M) bool all-ones
    (the cache stores no mask because lagernvs marks every token valid, geo_encoder.py:139)."""
    emb = data['geo_emb'].to(device)
    mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=device)
    return emb, mask


def geo_emb_from_raw_cache(geo_encoder, data, device):
    """[new 2026-08-29] pre-ln 캐시 경로 (cfg.geo_raw_cache_dir).

    `geo_emb_from_cache` 와 달리 **인코더를 통과시킨다** — 캐시가 담고 있는 건 DA3 backbone 의
    frozen 출력(ln 직전)이고, 그 위의 `ln` + `GeoEncoder.proj` 는 학습 대상이라 매 스텝 다시
    돌아야 한다. 얼린 구간이 frozen 이므로 값은 on-the-fly 와 비트 단위로 같다
    (실측 `geo_emb max|d| 0`, 단 `da3_cam_token_per_sample: true` 여야 한다).
    캐시는 fp32 로 굽는다 — `encode_raw` 가 bf16 autocast 아래서도 fp32 를 돌려주기 때문이다
    (`da3_geo_encoder.encode_raw` docstring). `.float()` 는 낡은 bf16 캐시가 섞여 들어와도
    on-the-fly 경로 (`feats[-1].float()`) 와 dtype 이 갈리지 않게 하는 보험이다."""
    return geo_encoder.from_raw(data['geo_raw'].to(device).float())


def geo_encode(geo_encoder, data, device):
    """Run the geo encoder on a batch. When cfg.geo_posed and the batch carries geo-view
    geometry, feed the posed lagernvs cam_token; otherwise unposed (cam_token=None).

    [new 2026-08-07] geo_encoder='custom' 은 images 말고도 픽셀 Plücker/log-depth/valid 가
    필요해서 backend 가 batch dict 을 통째로 받는다 (models/geo_encoder.py:_CustomBackend).
    wants_batch 로 분기하며, lagernvs 는 이 플래그가 False 라 아래 기존 경로 그대로다."""
    if getattr(geo_encoder, 'wants_batch', False):
        # [ablation 2026-08-10] custom_geo_channels 에 따라 필요한 키가 줄어든다
        # (no_depth -> logd/valid 없음, rgb_only -> Plücker 도 없음).
        need = tuple(getattr(geo_encoder, 'needs_keys',
                             ('images', 'geo_plucker_map', 'geo_logd', 'geo_valid')))
        miss = [k for k in need if k not in data]
        if miss:
            raise KeyError(f"geo_encoder='custom' needs {miss} in the batch — dataset_dl3dv 의 "
                           f"geo_custom 분기가 꺼져 있다 (geo_encoder/pose_source 확인)")
        return geo_encoder.forward_batch({k: data[k].to(device) for k in need})
    images = data['images'].to(device)
    cam_token = None
    if getattr(cfg, 'geo_posed', False) and 'geo_c2w' in data:
        # skip LagerNVS's own 1.35*ctx_max normalization -> reuse the target's initial
        # 1.35*max_dist (avg_scale) so context & target share one frame+scale.
        override = (data['avg_scale'].to(device)
                    if getattr(cfg, 'geo_lagernvs_skip_ctx_norm', False) else None)
        cam_token = geo_encoder.build_cam_token(
            data['geo_c2w'].to(device), data['geo_fxfycxcy'].to(device),
            data['geo_hw'].to(device), override_scale=override)
    return geo_encoder(images, cam_token)


def attach_geo_cam(emb, data, device):
    """[new] cfg.geo_cam_embed: append a raw camera embedding to every geo patch token.

      'relfirst' -- (B, V, 11) each context view's pose relative to the target's FIRST camera
                    (dataset_dl3dv._geo_cam_param), broadcast over that view's P patches.
      'plucker'  -- (B, V, P, 6) a Plücker ray PER patch token in the same frame/units
                    (dataset_dl3dv._geo_cam_plucker); no broadcast, one value per token.

      emb (B, M, C) with M = V*P  ->  (B, M, C + 11) or (B, M, C + 6)

    The raw dims ride along inside geo_emb so no call site's signature changes; the model splits
    them off and lifts them with a trainable MLP (CameraDiffusionModel._lift_geo_cam). No-op when
    geo_cam_embed is off or the batch carries no geo_cam_param."""
    if not getattr(cfg, 'geo_cam_embed', None) or 'geo_cam_param' not in data:
        return emb
    cam = data['geo_cam_param'].to(device).to(emb.dtype)   # (B,V,11) | (B,V,P,6)
    B, M, _ = emb.shape
    V = cam.shape[1]
    if M % V:
        raise ValueError(f"geo token count {M} not divisible by #views {V}")
    P = M // V
    if cam.dim() == 4:            # plucker: per-patch, must line up with the token grid
        if cam.shape[2] != P:
            raise ValueError(f"plucker grid {cam.shape[2]} != tokens per view {P} "
                             f"(M={M}, V={V}); check geo_image_hw vs _geo_patch_grid()")
        cam = cam.reshape(B, M, cam.shape[-1])
    else:                         # relfirst: per-view, broadcast over the view's patches
        cam = cam.unsqueeze(2).expand(B, V, P, cam.shape[-1]).reshape(B, M, cam.shape[-1])
    return torch.cat([emb, cam], dim=-1)


@torch.no_grad()
def geo_attn_probe(raw_model, scheduler, z, text_emb, text_mask, geo_emb, geo_mask, images=None,
                   timestep=500, n_views=None, grid_hw=None, make_figure=True):
    """[new 2026-08-07] geo cross-attention 을 학습 중에 추적한다. cfg.log_geo_attn 으로만 켜지고
    꺼져 있으면 호출되지 않으므로 기존 run 의 동작/속도는 그대로다.

    왜 attention weight 만으로는 부족한가: CrossAttention 은 norm(x + a) 이고 attention 은
    softmax 라 **행 합이 항상 1** 이다. 즉 "geo 를 얼마나 쓰는가"는 attention map 에 안 나온다.
    같이 재야 하는 게 resid_ratio = ||a|| / ||x|| — geo 가 hidden state 를 실제로 밀어내는 양이다.

    반환 dict
      resid_ratio_l{i}     layer i 의 ||a||/||x||  (+ 평균)
      attn_entropy         geo 토큰 M 개에 대한 attention 엔트로피(nats). ln(M) 이면 완전 균일
                           = 아무 patch 도 고르지 않는 상태.
      attn_entropy_norm    위를 ln(M) 으로 나눈 값 (1.0 = 균일). run 간 비교는 이걸로.
      view_mass_v{i}       M = V x P 로 접었을 때 view 별 attention 질량 (균일이면 1/V)
      view_max_over_uniform 가장 많이 보는 view 의 질량 / (1/V). 1.0 이면 view 구분을 안 한다.
      dpred_shuffle        geo 를 배치 축으로 roll(1) 했을 때 ||pred-pred_real||/||pred_real||.
                           **geo 를 실제로 쓰는지의 직접 지표**다 (토큰 통계는 그대로 두고 scene
                           짝만 깨므로, 0 이면 모델이 geo 내용을 안 본다).
      figure               (make_figure) V 개 context view 의 attention heatmap (+ 있으면 RGB)
    """
    import numpy as np
    probe = {}

    def _mk_hook(li):
        def hook(mod, inputs, output):
            x, (a, w) = inputs[0], (output[0], output[1])
            d = probe.setdefault(li, {})
            d['resid'] = (a.float().norm(dim=-1) / x.float().norm(dim=-1).clamp(min=1e-12)).mean().item()
            if w is not None:
                d['attn'] = w.detach().float()                     # (B,T,M) head 평균
        return hook

    handles = [raw_model.layers[li][5].attn.register_forward_hook(_mk_hook(li))
               for li in range(len(raw_model.layers))]
    try:
        B = z.shape[0]
        ts = torch.full((B,), int(timestep), device=z.device, dtype=torch.long)
        # 고정 seed noise — epoch 마다 같은 x_t 를 써야 지표가 epoch 간 비교 가능해진다
        g = torch.Generator(device='cpu').manual_seed(1234)
        noise = torch.randn(z.shape, generator=g).to(z.device)
        x_t = scheduler.add_noise(z, noise, ts)
        pred_real = raw_model(x_t, ts.float(), text_emb, text_mask, geo_emb, geo_mask)
        stats, resid = {}, []
        attn0 = None
        for li in sorted(probe):
            r = probe[li]['resid']
            stats[f'resid_ratio_l{li}'] = r
            resid.append(r)
            if attn0 is None:
                attn0 = probe[li].get('attn')
        stats['resid_ratio_mean'] = float(np.mean(resid)) if resid else float('nan')

        probe.clear()
        pred_sh = raw_model(x_t, ts.float(), text_emb, text_mask,
                            torch.roll(geo_emb, 1, 0), torch.roll(geo_mask, 1, 0))
        stats['dpred_shuffle'] = ((pred_sh - pred_real).norm() /
                                  pred_real.norm().clamp(min=1e-12)).item()

        fig = None
        if attn0 is not None:
            p = attn0.mean(dim=1)                                  # (B,M) target token 평균
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
            M = p.shape[-1]
            stats['attn_entropy'] = (-(p * p.clamp(min=1e-12).log()).sum(-1)).mean().item()
            stats['attn_entropy_norm'] = stats['attn_entropy'] / math.log(M)
            V = int(n_views) if n_views else 6
            if M % V == 0:
                P = M // V
                vm = p.view(-1, V, P).sum(-1).mean(0)              # (V,)
                for i in range(V):
                    stats[f'view_mass_v{i}'] = vm[i].item()
                stats['view_max_over_uniform'] = (vm.max().item() * V)
                if make_figure and grid_hw is not None:
                    fig = _geo_attn_figure(p[0].view(V, P), grid_hw, images, stats)
        return stats, fig
    finally:
        for h in handles:
            h.remove()


def _geo_attn_figure(attn_v, grid_hw, images, stats):
    """attn_v (V,P) -> V 열짜리 figure. images (B,V,3,H,W) 가 있으면 위에 RGB 를 같이 깐다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    V = attn_v.shape[0]
    a = attn_v.detach().float().cpu().numpy().reshape(V, gh, gw)
    has_rgb = images is not None
    rows = 2 if has_rgb else 1
    fig, axes = plt.subplots(rows, V, figsize=(2.4 * V, 2.6 * rows), squeeze=False)
    # vmin/vmax 를 **6개 view 전체**에서 잡는다. imshow 기본값은 패널마다 따로 정규화해서
    # view 간 밝기 비교가 무의미해진다 (view mass 차이가 그림에서 지워진다).
    vmin, vmax = float(a.min()), float(a.max())
    im_h = None
    for v in range(V):
        if has_rgb:
            im = images[0, v].detach().float().cpu().numpy().transpose(1, 2, 0)
            im = (im - im.min()) / max(im.max() - im.min(), 1e-8)
            axes[0][v].imshow(im)
            axes[0][v].set_title(f"view {v}" + (" (=target s)" if v == 0 else ""), fontsize=8)
            axes[0][v].axis('off')
        ax = axes[rows - 1][v]
        im_h = ax.imshow(a[v], cmap='inferno', vmin=vmin, vmax=vmax)
        ax.set_title(f"mass {a[v].sum():.4f}", fontsize=8)
        ax.axis('off')
    if im_h is not None:
        fig.colorbar(im_h, ax=axes[rows - 1, :].tolist(), fraction=0.02)
    # 그림 안 텍스트는 영문만 쓴다 — matplotlib 기본 폰트(DejaVu Sans)에 한글 glyph 가 없어
    # 한글을 넣으면 전부 두부(□)로 나온다.
    fig.suptitle(f"geo cross-attn (uniform = {1.0 / a.size:.2e}) | "
                 f"resid ||a||/||x|| {stats.get('resid_ratio_mean', float('nan')):.3f} | "
                 f"entropy_norm {stats.get('attn_entropy_norm', float('nan')):.4f} | "
                 f"dpred_shuffle {stats.get('dpred_shuffle', float('nan')):.4f}", fontsize=9)
    return fig


def build_track_cond(data, t_lat, device, dropout_p=0.0):
    """[new 2026-08-27] data['target_track'] (B,T,4)[xyz,valid] -> (B,t_lat,4) concat 조건.

    키가 없으면 None (기존 arm 경로 그대로). VAE 가 49프레임을 stride-2 두 번으로 t_lat 토큰으로
    줄이므로 조건도 같은 축으로 linear interp 한다. null = 전 채널 0 (valid 포함) — concat 조건은
    uncond forward 에도 채널이 있어야 해서 CFG 를 '빼기'가 아니라 null 값으로 정의한다.
    dropout_p 는 per-sample: 추론에서 track 없이 쓸 수 있게 null 조건을 학습에 남겨 두는 장치.
    """
    tt = data.get('target_track')
    if tt is None:
        return None
    tt = tt.to(device).float().transpose(1, 2)                      # (B,4,T)
    cond = F.interpolate(tt, size=int(t_lat), mode='linear',
                         align_corners=True).transpose(1, 2)        # (B,t_lat,4)
    if dropout_p > 0:
        drop = torch.rand(cond.shape[0], device=device) < float(dropout_p)
        cond[drop] = 0.0
    return cond


def build_video_cond(data, device):
    """[new 2026-09-03] PE-AV 스트림(D117 video CA) 을 모델 kwargs dict 로 만든다.

    데이터셋이 `peav_video` 를 안 실어 주면 **빈 dict** 라 model(...) 호출이 글자 그대로
    예전과 같아진다 (video_latent_dim=0 인 arm 은 이 경로 자체가 no-op).
    `peav_text` 는 두 arm 에서 쓰임이 다르다 — arm A 는 video CA 의 key/value 앞쪽에 붙고
    (`video_text_dim>0`), arm B 는 umt5 를 대신해 **text CA** 로 들어간다. 그 갈래는
    `cfg.video_text_in_stream` 이 정하고, 여기서는 arm A 일 때만 kwargs 에 담는다."""
    kw = {}
    if 'peav_video' in data:
        kw['video_emb'] = data['peav_video'].to(device).float()
    if kw and getattr(cfg, 'video_text_in_stream', False) and 'peav_text' in data:
        kw['video_text_emb'] = data['peav_text'].to(device).float()
        kw['video_text_mask'] = data['peav_text_mask'].to(device).bool()
    return kw


@torch.no_grad()
def sample(model, scheduler, traj_len, text_emb, text_masks, point_emb, point_mask, generator=None,
           cond=None, video_kw=None):
    """generator: x_T 추첨용 **CPU** torch.Generator. None 이면 전역 RNG (기존 동작).

    이걸 넘기면 sampling 이 완전히 결정적이 된다 — cfg.sampling_type='ddim' 의 DDIMScheduler
    는 eta=0 이라 step() 이 노이즈를 안 뽑으므로 확률적 요소가 x_T 하나뿐이기 때문이다.
    ('ddpm' 으로 바꾸면 step() 이 전역 RNG 에서 다시 뽑으므로 이 보장이 깨진다.)
    CPU generator 로 뽑아서 .to(device) 하는 이유는 GPU RNG 가 device/커널에 따라 다른 수열을
    내서, 같은 시드라도 GPU 를 바꾸면 다른 표본이 나오기 때문이다."""
    B = text_emb.size(0)
    device = text_emb.device
    x_t = torch.randn(B, traj_len, cfg.cam_dim, generator=generator).to(device)

    scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)
    for t in scheduler.timesteps:
        timesteps = torch.full(
            (B,),
            t,
            device=device,
        )
        noise_pred = model(x_t, timesteps.float(), text_emb, text_masks, point_emb, point_mask,
                           cond=cond, **(video_kw or {}))
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample
    return x_t


from tau_sampler import sample_tau


def per_token_flow_loss(model, z, text_emb, text_mask, geo_emb, geo_mask, return_stats=False):
    """Diffusion-Forcing rectified-flow loss on the causal-VAE latent z (B, W, D).
      eps~N(0,1); z_noisy=(1-tau)z+tau*eps; v_target=eps-z; model predicts v (per-token tau).
      tokens with tau < cfg.loss_tau_min are context -> excluded from loss.
    Batch-level 10% text CFG dropout (null embedding)."""
    B, W, _ = z.shape
    device = z.device
    eps = torch.randn_like(z)
    tau, labels = sample_tau(B, W, device)                       # (B,W), (B,)
    tau_e = tau.unsqueeze(-1)
    z_noisy = (1 - tau_e) * z + tau_e * eps
    v_target = eps - z
    if torch.rand(1).item() < getattr(cfg, 'cfg_dropout_p', 0.1):  # batch-level CFG dropout
        text_emb = torch.zeros_like(text_emb)
        text_mask = torch.ones_like(text_mask)
    v_pred = model(z_noisy, tau * 1000.0, text_emb, text_mask, geo_emb, geo_mask)  # per-token time
    mask = (tau >= getattr(cfg, 'loss_tau_min', 0.02)).float()   # (B,W)
    per_tok = ((v_pred - v_target) ** 2).mean(-1)                 # (B,W)
    denom = mask.sum().clamp(min=1.0)                            # zero-denom guard
    loss = (mask * per_tok).sum() / denom
    if return_stats:
        return loss, tau.detach(), labels.detach(), per_tok.detach()
    return loss


def _ar_chunk_bounds(T, chunk_size):
    """[(s,e), ...] covering [0,T) in chunks of chunk_size (last may be smaller)."""
    bounds, s = [], 0
    while s < T:
        bounds.append((s, min(s + chunk_size, T)))
        s += chunk_size
    return bounds


def ar_train_loss(raw_model, scheduler, traj_latents, text_emb, text_mask, geo_emb, geo_mask, chunk_size):
    """Chunk-wise AR DDPM loss: for each chunk, past chunks are CLEAN (teacher-forcing),
    current chunk gets fresh DDPM noise at a per-chunk timestep; causal self-attn conditions
    the current chunk on the clean past + text + geo. Loss = mean MSE(pred_noise, noise)."""
    B, T, _ = traj_latents.shape
    device = traj_latents.device
    total, n = 0.0, 0
    for s, e in _ar_chunk_bounds(T, chunk_size):
        cur = traj_latents[:, s:e]
        history = traj_latents[:, :s] if s > 0 else None
        noise = torch.randn_like(cur)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy = scheduler.add_noise(cur, noise, t)
        pred = raw_model.forward_ar(noisy, t.float(), text_emb, text_mask, geo_emb, geo_mask, history=history)
        total = total + F.mse_loss(pred, noise)
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def ar_sample(raw_model, scheduler, traj_len, text_emb, text_masks, geo_emb, geo_mask, chunk_size, cam_dim):
    """Autoregressive generation: denoise chunk 0 from noise, then each next chunk
    conditioned on the previously generated (clean) chunks via causal self-attn."""
    B, device = text_emb.size(0), text_emb.device
    scheduler.set_timesteps(num_inference_steps=cfg.diffusion_inference_step, device=device)
    generated = None
    for s, e in _ar_chunk_bounds(traj_len, chunk_size):
        x = torch.randn(B, e - s, cam_dim, device=device)
        for tt in scheduler.timesteps:
            t = torch.full((B,), tt, device=device).float()
            noise_pred = raw_model.forward_ar(x, t, text_emb, text_masks, geo_emb, geo_mask, history=generated)
            x = scheduler.step(noise_pred, tt, x).prev_sample
        generated = x if generated is None else torch.cat([generated, x], dim=1)
    return generated

def train():
    T = cfg.diffusion_max_step 
    traj_len = cfg.num_cam
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    # ---- resume: peek checkpoint. full = {model,opt,global_step,epoch,best_val,...};
    #      a pure state_dict (old last.pth) -> weights-only warm-start. ----
    _resume = None
    if getattr(cfg, 'load_ckpt_path', None):
        _resume = torch.load(cfg.load_ckpt_path, map_location='cpu')
        if not (isinstance(_resume, dict) and 'model' in _resume):
            _resume = {'model': _resume, '_weights_only': True}
    _full_resume = _resume is not None and not _resume.get('_weights_only', False)

    if accelerator.is_main_process:
        if _full_resume and _resume.get('exp_name'):
            exp_name = _resume['exp_name']                 # reuse result dir + wandb run
        else:
            exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_{cfg.exp_name}'
    else:
        exp_name = None
    exp_name = broadcast_object_list([exp_name])[0]
    result_dir = os.path.abspath(os.path.join('../results', exp_name))
    os.makedirs(result_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_data_dir = os.path.join(result_dir, 'test')
    os.makedirs(eval_data_dir, exist_ok=True)

    _wandb_init = {"name": f"{exp_name}"}
    if _full_resume and _resume.get('wandb_id'):           # continue the same wandb run
        _wandb_init = {"id": _resume['wandb_id'], "resume": "allow", "name": f"{exp_name}"}
    accelerator.init_trackers(
        project_name="camera-diffusion-training",
        config=cfg_dict,
        init_kwargs={"wandb": _wandb_init},
    )
    try:
        _wandb_id = accelerator.get_tracker("wandb", unwrap=True).id
    except Exception:
        _wandb_id = None

    if accelerator.is_main_process:            # full resolved config as YAML (run + wandb dir)
        from hydra_cfg import save_cfg_yaml
        save_cfg_yaml(cfg_dict, os.path.join(result_dir, 'config.yaml'))
        try:
            save_cfg_yaml(cfg_dict, os.path.join(wandb.run.dir, 'config.yaml'))
        except Exception as _e:
            print(f"(train) wandb config.yaml save skipped: {_e}")

    trainer = Trainer(cfg=cfg)
    trainer._make_batch_generator()
    # NOTE: trainer._make_model() removed — it allocated an unused DataParallel model
    # on GPU (get_model() defaults to cam_dim=11); training builds its own `model` below.

    if cfg.sampling_type == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=T,
            prediction_type=cfg.prediction_type,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )
    elif cfg.sampling_type == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=T,
            prediction_type=cfg.prediction_type,
            beta_schedule=cfg.beta_schedule,
            clip_sample=cfg.clip_sample,
            set_alpha_to_one=cfg.set_alpha_to_one,
            steps_offset=cfg.steps_offset,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
        )    

    pc_encoder = PCEncoder().to(device) if PCEncoder is not None else None
    # Geo encoder (image-based, frozen, on-the-fly). Feeds the latent model's geo latent.
    # TODO(C-4): switch the training loop below to compute geo_emb/geo_mask via
    #   geo_emb, geo_mask = geo_encoder(images, cam_token)
    # (needs the dataset to provide multi-view images) instead of pc_embeds/pc_masks.
    geo_encoder = build_geo_encoder(cfg).to(device) if getattr(cfg, 'geo_encoder', None) else None
    if cfg.text_encoder == 'T5':
        text_encoder = T5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=device,
            checkpoint_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_checkpoint_path),
            tokenizer_path=os.path.join(cfg.t5_checkpoint_dir, cfg.t5_tokenizer_path),
            shard_fn=None)
    elif cfg.text_encoder == 'CLIP':
        from core_pkg.common.utils.clip_utils import get_model, encode_text_clip
        text_encoder, tokenizer = get_model(cfg.clip_repo_name, device=device)
    
    # [new] cfg.geo_cam_embed widens the geo cross-attention input: geo_proj sees
    # geo_latent_dim + geo_cam_embed_dim instead of geo_latent_dim. Off (0) -> identical model.
    _geo_kw = {}
    if getattr(cfg, 'geo_cam_embed', None):
        # relfirst = 11 raw dims per view (rot6d + trans + fx/2cx, fy/2cy);
        # plucker  = 6 raw dims per patch token (direction + moment).
        _raw = 6 if cfg.geo_cam_embed == 'plucker' else 11
        _geo_kw = dict(geo_latent_dim=getattr(cfg, 'geo_latent_dim', 768),
                       geo_cam_raw_dim=_raw,
                       geo_cam_embed_dim=getattr(cfg, 'geo_cam_embed_dim', 128))
        print(f"(model) geo_cam_embed={cfg.geo_cam_embed}: geo_proj input = "
              f"{_geo_kw['geo_latent_dim']} + {_geo_kw['geo_cam_embed_dim']}")
    # [new 2026-08-27] cfg.target_track_dim>0: subject OBB track 을 x_t 채널에 concat.
    # 0 (기본) 이면 cond_dim=0 -> cam_in 이 기존과 동일 (state_dict/동작 비트 동일).
    _track_dim = int(getattr(cfg, 'target_track_dim', 0) or 0)
    if _track_dim > 0:
        assert not getattr(cfg, 'is_ar', False) and not getattr(cfg, 'per_token_noise', False), \
            "target_track_dim>0 은 표준 diffusion 경로에만 배선되어 있다 (is_ar/per_token_noise 미지원)"
        # dropout / val 조건은 로그에 안 남으면 나중에 run 을 봐도 어느 쪽인지 알 수 없다.
        print(f"(model) target_track_dim={_track_dim}: cam_in input = cam_dim + {_track_dim} | "
              f"train dropout={float(getattr(cfg, 'target_track_dropout', 0.1))} | "
              f"val cond={'null (track 없이)' if getattr(cfg, 'target_track_val_drop', False) else 'track'}")
    # [new 2026-09-03] D117 video CA. cfg.video_latent_dim=0 (기본) 이면 _vid_kw 가 비어 있어
    # 모델 구조·state_dict 가 예전과 비트 동일하다. 1792 = PE-AV(pe-av-large) visual tower 의
    # per-frame 토큰 차원. video_text_in_stream (arm A) 이면 PE-AV text(1024) 가 같은 CA 의
    # key/value 앞쪽에 붙는다.
    _vid_kw = {}
    _vld = int(getattr(cfg, 'video_latent_dim', 0) or 0)
    if _vld > 0:
        assert not getattr(cfg, 'is_ar', False) and not getattr(cfg, 'per_token_noise', False), \
            "video_latent_dim>0 은 표준 diffusion 경로에만 배선되어 있다 (is_ar/per_token_noise 미지원)"
        _vtd = 1024 if getattr(cfg, 'video_text_in_stream', False) else 0
        _vid_kw = dict(video_latent_dim=_vld, video_text_dim=_vtd,
                       peav_in_ln=bool(getattr(cfg, 'peav_in_ln', True)),
                       video_gate=bool(getattr(cfg, 'video_gate', True)))
        print(f"(model) video CA: video_latent_dim={_vld} video_text_dim={_vtd} "
              f"in_ln={_vid_kw['peav_in_ln']} gate={_vid_kw['video_gate']} "
              f"(순서 text CA -> video CA -> geo CA)")
    # arm B (text_encoder='PEAV') 는 text CA 입력이 umt5 4096 이 아니라 PE-AV 1024 다.
    if cfg.text_encoder == 'PEAV':
        _geo_kw['text_dim'] = 1024
        # [new 2026-09-03 / FIX] PE-AV text 는 std 95 · absmax 12928 (umt5 는 ~unit) 이라
        # text_proj 앞에 LayerNorm 이 필요하다. umt5 arm 은 text_in_ln=False 라 무영향.
        _geo_kw['text_in_ln'] = bool(getattr(cfg, 'peav_in_ln', True))
        print(f"(model) text_encoder=PEAV: text_proj 입력 = 1024 (umt5 미사용) "
              f"in_ln={_geo_kw['text_in_ln']}")
    if cfg.point_encoder != 'custom':
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim, cond_dim=_track_dim, **_geo_kw, **_vid_kw)
    else:
        model = CameraDiffusionModel(cam_dim=cfg.cam_dim, cond_dim=_track_dim,
                                     pc_encoder=pc_encoder, **_geo_kw, **_vid_kw)
    
    # Load weights from the (peeked) resume checkpoint. Optimizer/step/epoch are
    # restored after accelerator.prepare() below (full resume only).
    if _resume is not None:
        _mm, _uu = model.load_state_dict(_resume['model'], strict=False)
        _kind = "full-resume" if _full_resume else "weights warm-start"
        print(f"(resume) {_kind} model from {cfg.load_ckpt_path} "
              f"(missing={len(_mm)}, unexpected={len(_uu)})")

    model.train()
    # [new 2026-08-07] geo_encoder='custom' 은 frozen 이 아니다 (안쪽 DINOv2 만 frozen). 그
    # 파라미터를 optimizer 에 넣지 않으면 GeoTokenizer/ln/proj 가 랜덤 초기값에 영원히 머문다.
    # lagernvs/scenetok 은 trainable=False 라 여기서 아무 것도 달라지지 않는다.
    _geo_trainable = bool(geo_encoder is not None and getattr(geo_encoder, 'trainable', False))
    # [new 2026-08-07] geo attention map 을 view 별 patch 격자로 되접을 때 쓰는 (gh, gw).
    # custom / da3 backend 만 격자가 확정적이다 (입력 hw / patch). 다른 backend 는
    # None -> geo_attn_probe 가 스칼라 지표만 내고 figure 는 안 만든다.
    _geo_grid_hw = None
    _geo_name = str(getattr(cfg, 'geo_encoder', None))
    if _geo_name in ('custom', 'da3'):
        # da3 의 patch 는 DepthAnything3Net.PATCH_SIZE = 14 고정이다.
        _p = 14 if _geo_name == 'da3' else int(getattr(cfg, 'custom_geo_patch', 14))
        _ihw = getattr(cfg, 'da3_geo_input_hw' if _geo_name == 'da3'
                       else 'custom_geo_input_hw', None)
        if _ihw:
            _geo_grid_hw = (int(_ihw[0]) // _p, int(_ihw[1]) // _p)
    _geo_frozen_keys = set()
    if _geo_trainable:
        # .train() 을 부르지 않는다 — SceneEncoder.__init__ 이 frozen DINO 를 eval() 로 내려
        # 놨는데 부모에서 train() 을 부르면 재귀적으로 되돌아간다. GeoTokenizer 는 GroupNorm/
        # LayerNorm 뿐이라 train/eval 차이가 없어서 생성 직후 상태 그대로가 맞다.
        _geo_params = [p for p in geo_encoder.parameters() if p.requires_grad]
        # frozen 파라미터(=DINOv2 ViT-L, ~300 M)는 ckpt 에서 뺀다. 안 빼면 매 epoch 1.2 GB 를
        # 쓰는데, 그 값은 checkpoints/dinov2-large 에 이미 있고 절대 바뀌지 않는다.
        _geo_frozen_keys = {n for n, p in geo_encoder.named_parameters() if not p.requires_grad}
        _n = sum(p.numel() for p in _geo_params)
        print(f"(geo) trainable backend '{cfg.geo_encoder}': {len(_geo_params)} tensors / "
              f"{_n / 1e6:.2f} M params -> AdamW "
              f"(frozen {len(_geo_frozen_keys)} tensors excluded from ckpt)")
        opt = torch.optim.AdamW(list(model.parameters()) + _geo_params, lr=cfg.lr)
        _opt_param_names = ([n for n, _ in model.named_parameters()]
                            + [n for n, p in geo_encoder.named_parameters() if p.requires_grad])
        if _resume is not None and _resume.get('geo') is not None:
            _gm, _gu = geo_encoder.load_state_dict(_resume['geo'], strict=False)
            print(f"(resume) geo encoder from {cfg.load_ckpt_path} "
                  f"(missing={len(_gm)} [frozen DINO 포함], unexpected={len(_gu)})")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        _opt_param_names = [n for n, _ in model.named_parameters()]

    if cfg.use_vae:
        if getattr(cfg, 'causal_vae', False):
            # AR: causal VAE (past-only conv) — reuse CamVLA causal_vae_v1 ckpt (49->13, 11-dim)
            from types import SimpleNamespace
            from models.cam_causal_vae import CameraVAE as CausalCameraVAE
            camera_vae = CausalCameraVAE(SimpleNamespace(
                in_dim=11, out_dim=11, hidden_size=64, latent_dim=cfg.cam_dim,
                scale_factor=cfg.vae_latent_scale)).to(device)
        else:
            camera_vae = CameraVAE(latent_dim=cfg.cam_dim).to(device)
        camera_vae.load_state_dict(torch.load(cfg.vae_ckpt_path, map_location=device))
        camera_vae.eval()

    # For evaluation (CLaTr) — optional
    clip_model = load_clip_model(cfg.clip_version, device=device) if _HAS_CLIP else None

    global_step = 0
    start_epoch = 0

    model, opt, train_dataloader, valid_dataloader = accelerator.prepare(
        model, opt, trainer.batch_generator, trainer.valid_batch_generator
    )

    best_val = float('inf')   # best val/loss_traj so far (for best.pth)
    if _full_resume:          # restore optimizer + counters for seamless continuation
        # [fix 2026-08-10] AdamW state_dict 는 param 을 **인덱스**로만 참조한다. 그래서 저장 이후
        # nn.Module 의 **등록 순서**가 바뀌면 (파라미터 개수가 같아도) 모멘트가 엉뚱한 param 에
        # 실린다. 실제로 sd_whuman_customgeo (fkfqww00) 재개가 이렇게 죽었다: 커밋 916bc7a 가
        # SceneEncoder.__init__ 에서 self.ln_d 를 self.geo 앞으로 옮기면서 geo 22개 중
        # GeoTokenizer 16개가 2칸 밀렸고, opt.step() 에서
        #   RuntimeError: The size of tensor a (14) must match the size of tensor b (1024)
        # 가 났다 (ray.weight 의 exp_avg 가 ln_d.weight 의 grad 와 짝지어짐).
        # -> 이제 이름 목록을 같이 저장하고, 재개 시 이름 기준으로 인덱스를 다시 맞춘다.
        _opt_sd = _resume['opt']
        _saved_names = _resume.get('opt_param_names')
        if _saved_names is None and _resume.get('model') is not None:
            # 구 ckpt 호환: 저장 당시 순서 = model state_dict 키 순서 + geo state_dict 키 순서.
            # (opt 은 list(model.parameters()) + geo trainable params 로 만들어지고, 두 sd 모두
            #  등록 순서를 그대로 따른다. 개수가 안 맞으면 buffer 가 섞인 것이니 포기한다.)
            _cand = list(_resume['model'].keys()) + list((_resume.get('geo') or {}).keys())
            if len(_cand) == len(_opt_param_names):
                _saved_names = _cand
        if _saved_names is not None and list(_saved_names) != list(_opt_param_names):
            if sorted(_saved_names) != sorted(_opt_param_names):
                raise RuntimeError(
                    "resume: optimizer param 이름 집합이 ckpt 와 다르다 — 모델 구조가 바뀐 "
                    f"체크포인트다. ckpt {len(_saved_names)}개 / 현재 {len(_opt_param_names)}개, "
                    f"ckpt 에만 있는 것 {sorted(set(_saved_names) - set(_opt_param_names))[:5]}, "
                    f"현재에만 있는 것 {sorted(set(_opt_param_names) - set(_saved_names))[:5]}")
            _new_of = {n: i for i, n in enumerate(_opt_param_names)}
            _remap = {i: _new_of[n] for i, n in enumerate(_saved_names)}
            _opt_sd = {
                'state': {_remap[int(i)]: s for i, s in _opt_sd['state'].items()},
                'param_groups': [dict(g, params=sorted(_remap[int(i)] for i in g['params']))
                                 for g in _opt_sd['param_groups']],
            }
            _moved = sum(1 for i, j in _remap.items() if i != j)
            print(f"(resume) optimizer param 순서가 ckpt 와 달라 이름 기준으로 재매핑했다 "
                  f"({_moved}/{len(_remap)} 개 이동).")
        opt.load_state_dict(_opt_sd)
        # 재매핑이 맞았는지 shape 로 확인 (틀리면 opt.step() 에서야 터진다).
        _flat = [p for g in getattr(opt, 'optimizer', opt).param_groups for p in g['params']]
        for _i, _p in enumerate(_flat):
            _s = getattr(opt, 'optimizer', opt).state.get(_p, {})
            if 'exp_avg' in _s and tuple(_s['exp_avg'].shape) != tuple(_p.shape):
                raise RuntimeError(f"resume: optimizer state shape 불일치 (param {_i} "
                                   f"{_opt_param_names[_i]}: param {tuple(_p.shape)} vs "
                                   f"exp_avg {tuple(_s['exp_avg'].shape)})")
        _real_opt = getattr(opt, 'optimizer', opt)
        for _st in _real_opt.state.values():              # move AdamW moments to device
            for _k, _v in _st.items():
                if torch.is_tensor(_v):
                    _st[_k] = _v.to(device)
        global_step = int(_resume.get('global_step', 0))
        start_epoch = int(_resume.get('epoch', -1)) + 1
        best_val = float(_resume.get('best_val', float('inf')))
        print(f"(resume) continuing at epoch {start_epoch}, global_step {global_step}, "
              f"best_val {best_val:.6f}")

    def run_validation(epoch, global_step):
        """Per-epoch validation: sample trajectories, log val losses (+ optional
        CLaTr metrics) to wandb, and keep only last.pth + best.pth (by val/loss_traj)."""
        nonlocal best_val
        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_dataloader)
            total_loss_latent = torch.tensor(0.0, device=accelerator.device)
            total_loss_traj = torch.tensor(0.0, device=accelerator.device)
            total_samples = torch.tensor(0, device=accelerator.device)
            data_name_list = []
            _pt = getattr(cfg, 'per_token_noise', False)   # Diffusion-Forcing val monitoring
            tau_bin_sum = torch.zeros(10, device=accelerator.device)
            tau_bin_cnt = torch.zeros(10, device=accelerator.device)
            pat_sum = torch.zeros(4, device=accelerator.device)
            pat_cnt = torch.zeros(4, device=accelerator.device)
            for step, data in enumerate(pbar):
                if step >= cfg.val_max_batches:   # cap validation cost
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

                if cfg.load_points:
                    if 'pc' in data:
                        point = data['pc']
                        if cfg.point_encoder != 'custom':
                            pc_embeds, pc_masks, point_obj = pc_encoder(point)
                        else:
                            pc_embeds, pc_masks, point_obj = model.pc_encoder(point)
                    elif 'pc_embeds' in data:
                        pc_embeds = data['pc_embeds'].to(device)
                        pc_masks = data['pc_masks'].to(device)

                else:
                    pc_embeds, pc_masks = None, None
                # Geo latent conditioning (image-based, frozen) — mirrors the train loop.
                if 'geo_raw' in data:
                    with accelerator.autocast():
                        pc_embeds, pc_masks = geo_emb_from_raw_cache(geo_encoder, data, device)
                elif 'geo_emb' in data:
                    pc_embeds, pc_masks = geo_emb_from_cache(data, device)
                elif geo_encoder is not None and 'images' in data:
                    # train loop 과 같은 autocast 를 걸어 val 쪽 geo 토큰 수치가 어긋나지 않게 한다
                    # (frozen backend 는 원래도 여기서 fp32 였으므로 그대로 둔다).
                    if _geo_trainable:
                        with accelerator.autocast():
                            pc_embeds, pc_masks = geo_encode(geo_encoder, data, device)
                    else:
                        pc_embeds, pc_masks = geo_encode(geo_encoder, data, device)
                if pc_embeds is not None:
                    pc_embeds = attach_geo_cam(pc_embeds, data, device)
                if cfg.text_encoder == 'T5':
                    text_embeds, text_masks = text_encoder(text_prompt, device)
                elif cfg.text_encoder == 'CLIP':
                    text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
                elif cfg.text_encoder == 'PEAV':
                    # [new 2026-09-03] arm B: umt5 대신 PE-AV text tower (1024-d) 를 text CA 로.
                    # 인코더를 여기서 돌리지 않고 데이터셋 캐시를 그대로 쓴다 (frozen + 캡션이
                    # 세그먼트 상수라 매 스텝 forward 할 이유가 없다). model 의 text_dim 도
                    # 1024 여야 한다 — 아래 모델 생성부가 cfg.text_encoder 를 보고 맞춘다.
                    text_embeds, text_masks = data['peav_text'].to(device), data['peav_text_mask'].to(device)
                text_embeds = text_embeds.float()
                text_masks = text_masks.bool()
                # [new 2026-09-03] video CA kwargs. 캐시가 없으면 빈 dict = 기존 호출.
                _vkw = build_video_cond(data, device)
                if cfg.use_vae:
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                else:
                    traj_latents = traj

                # [new 2026-08-07] geo cross-attention tracking. cfg.log_geo_attn 이 꺼져 있으면
                # (기본값) 아예 안 돈다 -> 기존 run 의 동작·속도 그대로. 첫 배치 하나에서만 재고
                # forward 2회 추가라 비용은 무시할 수준이다.
                if (step == 0 and getattr(cfg, 'log_geo_attn', False) and pc_embeds is not None
                        and not _pt and not getattr(cfg, 'is_ar', False)):
                    _every = int(getattr(cfg, 'log_geo_attn_every', 5))
                    _fig_ok = (epoch % _every == 0)
                    try:
                        _gs, _gfig = geo_attn_probe(
                            accelerator.unwrap_model(model), noise_scheduler, traj_latents,
                            text_embeds, text_masks, pc_embeds, pc_masks,
                            images=data.get('images'),
                            timestep=int(getattr(cfg, 'log_geo_attn_timestep', 500)),
                            n_views=int(data['images'].shape[1]) if 'images' in data else None,
                            grid_hw=_geo_grid_hw, make_figure=_fig_ok)
                        accelerator.log({f"val/geo/{k}": v for k, v in _gs.items()}, step=global_step)
                        if _gfig is not None:
                            accelerator.log({"val/geo/attn_map": wandb.Image(_gfig)}, step=global_step)
                            import matplotlib.pyplot as _plt
                            _plt.close(_gfig)
                    except Exception as _e:      # 계측 실패가 학습을 죽이면 안 된다
                        print(f"[geo attn] probe 실패 (무시하고 계속): {type(_e).__name__}: {_e}")

                if _pt:   # per-token flow: masked val loss + tau-bin/pattern monitoring
                    vloss, tau, labels, per_tok = per_token_flow_loss(
                        model, traj_latents, text_embeds, text_masks, pc_embeds, pc_masks, return_stats=True)
                    total_loss_traj += vloss * B
                    total_samples += B
                    bins = (tau * 10).long().clamp(0, 9)
                    for bi in range(10):
                        m = bins == bi
                        if m.any():
                            tau_bin_sum[bi] += per_tok[m].sum(); tau_bin_cnt[bi] += m.sum()
                    for p in range(4):
                        m = labels == p
                        if m.any():
                            pat_sum[p] += per_tok[m].mean(); pat_cnt[p] += 1
                    continue

                if getattr(cfg, 'is_ar', False):
                    out = ar_sample(accelerator.unwrap_model(model), noise_scheduler, traj_len,
                                    text_embeds, text_masks, pc_embeds, pc_masks, cfg.ar_chunk_size, cfg.cam_dim)
                else:
                    # [new 2026-08-09] cfg.val_sample_seed 가 있으면 배치마다 (seed + step) 으로
                    # 시드한 전용 generator 로 x_T 를 뽑는다. epoch 간/arm 간 **같은 노이즈**를
                    # 쓰는 짝지은 비교가 되어, 가중치 변화만 val 곡선에 남는다. 실측 노이즈 폭은
                    # 160-segment val 에서 sd 0.0046 (plateau 평균의 7%) 이라 arm 격차를 삼킨다.
                    # 배치 index 를 더하는 이유: 전역 RNG 를 안 쓰므로 앞 배치의 소비량이나
                    # 다른 arm 의 RNG 사용 패턴에 흔들리지 않는다. batch_size 를 바꾸면 segment
                    # 와 노이즈의 짝이 달라지므로 비교하려는 run 끼리는 batch_size 를 맞춰야 한다.
                    # None (기본값 아님, 명시적 null) 이면 기존처럼 전역 RNG 를 쓴다.
                    _vs = getattr(cfg, 'val_sample_seed', None)
                    _g = torch.Generator().manual_seed(int(_vs) + step) if _vs is not None else None
                    # val 은 dropout 없이 실제 조건 그대로 (target_track_dim=0 이면 None = 기존).
                    # [new 2026-08-28] cfg.target_track_val_drop=true 면 val 을 **track 없이**
                    # (cond=None -> 모델이 전 채널 0 = null 을 채운다, forward:171) 돌린다.
                    # 학습 때 target_track_dropout 이 남겨 둔 그 조건이라 미학습 입력이 아니다.
                    _cond = (None if getattr(cfg, 'target_track_val_drop', False)
                             else build_track_cond(data, traj_len, device))
                    out = sample(model, noise_scheduler, traj_len, text_embeds, text_masks,
                                 pc_embeds, pc_masks, generator=_g, cond=_cond, video_kw=_vkw)

                val_loss_latent = F.mse_loss(out, traj_latents, reduction='mean')
                total_loss_latent += val_loss_latent * B

                if cfg.use_vae:
                    traj_pred = camera_vae.decode(out * cfg.vae_latent_scale)
                else:
                    traj_pred = out
                val_loss_traj = F.mse_loss(traj_pred, traj, reduction='mean')
                total_loss_traj += val_loss_traj * B
                total_samples += B

                ref_intrinsics = make_intrinsics(traj[:, :, -2:], width, height, intrinsics).cpu().tolist()
                pred_intrinsics = make_intrinsics(traj_pred[:, :, -2:], width, height, intrinsics).cpu().tolist()
                width = width.cpu().tolist()
                height = height.tolist()

                # anchor_pred_frame0: target 첫 카메라를 given 으로 (rel[0]=I 강제). 기본 False =
                # 기존 동작. GT 쪽은 이미 rel[0]=I 라 no-op 이지만 대칭을 위해 같이 넘긴다.
                _a0 = bool(getattr(cfg, 'anchor_pred_frame0', False))
                traj = out_to_trajectory(traj, scale, E0, device, anchor_frame0=_a0)
                traj_pred = out_to_trajectory(traj_pred, scale, E0, device, anchor_frame0=_a0)
                matrix_traj_ref = inverse_camera_matrix(traj)
                matrix_traj_pred = inverse_camera_matrix(traj_pred)
                matrix_traj_ref[:, :, :3, 1:3] *= -1
                matrix_traj_pred[:, :, :3, 1:3] *= -1
                matrix_traj_ref = matrix_traj_ref.cpu().tolist()
                matrix_traj_pred = matrix_traj_pred.cpu().tolist()

                if _HAS_CLIP and clip_model is not None:
                    seq_embeds, tok_embeds = encode_text(text_prompt, clip_model, max_token_length=None, device=device)
                    save_feats_custom(seq_embeds, data_name, Path(os.path.join(result_dir, 'seq')))
                    save_feats_custom(tok_embeds, data_name, Path(os.path.join(result_dir, 'token')))
                for i in range(len(data_name)):
                    text_path = os.path.join(eval_data_dir, f"{data_name[i]}_caption.json")
                    ref_path = os.path.join(eval_data_dir, f"{data_name[i]}_transforms_ref.json")
                    pred_path = os.path.join(eval_data_dir, f"{data_name[i]}_transforms_pred.json")
                    text_json = {
                        "Concise Interaction": text_prompt[i]
                    }
                    ref_json = {
                        "w": width[i][0],
                        "h": height[i][0],
                        "fl_x": ref_intrinsics[i][0][0][0],
                        "fl_y": ref_intrinsics[i][0][1][1],
                        "cx": ref_intrinsics[i][0][0][2],
                        "cy": ref_intrinsics[i][0][1][2],
                        "frames": [{
                            "transform_matrix": matrix_traj_ref[i][frame_idx],
                            "monst3r_im_id": frame_idx + 1}
                            for frame_idx in range(len(matrix_traj_ref[i]))]
                    }
                    pred_json = {
                        "w": width[i][0],
                        "h": height[i][0],
                        "fl_x": pred_intrinsics[i][0][0][0],
                        "fl_y": pred_intrinsics[i][0][1][1],
                        "cx": pred_intrinsics[i][0][0][2],
                        "cy": pred_intrinsics[i][0][1][2],
                        "frames": [{
                            "transform_matrix": matrix_traj_pred[i][frame_idx],
                            "monst3r_im_id": frame_idx + 1}
                            for frame_idx in range(len(matrix_traj_pred[i]))]
                    }
                    with open(text_path, 'w') as f:
                        json.dump(text_json, f, indent=4)
                    with open(ref_path, 'w') as f:
                        json.dump(ref_json, f, indent=4)
                    with open(pred_path, 'w') as f:
                        json.dump(pred_json, f, indent=4)
                data_name_list += data_name
            all_data_name_list = sorted(set(gather_object(data_name_list)))
            if accelerator.is_main_process:
                valid_txt_path = os.path.join(result_dir, 'test_valid.txt')
                with open(valid_txt_path, "w", encoding="utf-8") as f:
                    for name in all_data_name_list:
                        f.write(name + "\n")

                # CLaTr metric eval (subprocess) needs clip + hydra + the eval
                # subpackages; skipped when clip_extraction is unavailable or per-token
                # (per-token val saves no transforms; uses flow val loss + tau/pattern bins).
                if _HAS_CLIP and not _pt:
                    out, err, ret = run_command_in_dir([sys.executable, '-m', 'src.extraction', f'checkpoint_path={cfg.clatr_ckpt_path}', f'data_dir={result_dir}'], \
                                                    'evaluate/CLaTr')
                    print(out)
                    print(err)
                    out, err, ret = run_command_in_dir([sys.executable, '-m', 'src.eval_only', '--pred_path', f"{os.path.join(result_dir, 'preds.npy')}"], \
                                                    'evaluate/eval')
                    print(out)
                    print(err)
                    # robust: if the CLaTr subprocess failed, don't crash training
                    metric_json_path = os.path.join(result_dir, 'metrics.json')
                    if os.path.exists(metric_json_path):
                        with open(metric_json_path, 'r') as f:
                            metrics = json.load(f)
                        accelerator.log(metrics, step=global_step,)
                    else:
                        print("(CLaTr) metrics.json not produced; skipping CLaTr log this eval")

            total_loss_latent = accelerator.reduce(total_loss_latent, reduction='sum')
            total_loss_traj = accelerator.reduce(total_loss_traj, reduction='sum')
            total_samples = accelerator.reduce(total_samples, reduction='sum')
            print("valid samples:", total_samples)
            val_loss_latent_mean = (total_loss_latent / total_samples).item()
            val_loss_traj_mean = (total_loss_traj / total_samples).item()
            accelerator.log(
                {"val/loss_latent": val_loss_latent_mean,
                "val/loss_traj": val_loss_traj_mean},
                step=global_step,
            )
            if _pt:   # tau-bin + pattern val loss (Diffusion-Forcing monitoring)
                tbs = accelerator.reduce(tau_bin_sum, 'sum'); tbc = accelerator.reduce(tau_bin_cnt, 'sum')
                ps = accelerator.reduce(pat_sum, 'sum'); pc = accelerator.reduce(pat_cnt, 'sum')
                if accelerator.is_main_process:
                    logd = {f"val/tau_bin_{bi}": (tbs[bi] / tbc[bi]).item() for bi in range(10) if tbc[bi] > 0}
                    for p, nm in enumerate(['iid', 'ramp', 'bootup', 'full']):
                        if pc[p] > 0:
                            logd[f"val/pat_{nm}"] = (ps[p] / pc[p]).item()
                    accelerator.log(logd, step=global_step)

        # Keep only 2 checkpoints: last.pth (always) + best.pth (lowest val/loss_traj).
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
            # [new 2026-08-07] geo_encoder='custom' 의 학습된 가중치. last.pth/best.pth 는
            # "CameraDiffusionModel state_dict 하나"라는 계약을 그대로 두고 (eval_testset 등
            # 기존 소비자가 그대로 동작해야 한다) 옆에 별도 파일로 남긴다. frozen backend 에서는
            # 아무 파일도 생기지 않아 기존 run 디렉토리 구조와 동일하다.
            if _geo_trainable:
                _geo_sd = {k: v for k, v in geo_encoder.state_dict().items()
                           if k not in _geo_frozen_keys}
                torch.save(_geo_sd, os.path.join(ckpt_dir, "last_geo.pth"))
            if val_loss_traj_mean < best_val:
                best_val = val_loss_traj_mean
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
                if _geo_trainable:
                    torch.save(_geo_sd, os.path.join(ckpt_dir, "best_geo.pth"))
                print(f"(ckpt) new best val/loss_traj={best_val:.6f} @ epoch {epoch} -> best.pth")
            # [new 2026-08-08] cfg.ckpt_at_epochs 에 든 epoch 은 별도로 박아 둔다. best/last 는
            # 계속 덮어써지므로 "학습 중간 지점의 모델"을 나중에 볼 방법이 없다. 기본값이 빈
            # 리스트라 켜지 않으면 기존 run 디렉토리 구조와 동일하다.
            # [new 2026-08-09] cfg.ckpt_every_epochs 는 그 주기마다 자동으로 같은 일을 한다.
            # 파일명 규약(epoch<idx>.pth, idx = 0-based 루프 변수)은 ckpt_at_epochs 와 동일하게
            # 두어서 두 경로가 같은 epoch 에 대해 같은 파일을 가리킨다. epoch 0 은 건너뛴다.
            _marks = list(getattr(cfg, 'ckpt_at_epochs', None) or [])
            _every = getattr(cfg, 'ckpt_every_epochs', None)
            if _every and epoch > 0 and epoch % int(_every) == 0:
                _marks.append(epoch)
            for _e in dict.fromkeys(int(x) for x in _marks):      # 중복 제거, 순서 유지
                if epoch == _e:
                    torch.save(unwrapped_model.state_dict(),
                               os.path.join(ckpt_dir, f"epoch{_e}.pth"))
                    if _geo_trainable:
                        torch.save(_geo_sd, os.path.join(ckpt_dir, f"epoch{_e}_geo.pth"))
                    print(f"(ckpt) epoch {epoch} 고정 저장 -> epoch{_e}.pth")
            # full checkpoint for seamless resume (model+opt+step+epoch+best_val+ids)
            torch.save({
                'model': unwrapped_model.state_dict(),
                'geo': _geo_sd if _geo_trainable else None,
                'opt': opt.state_dict(),
                # opt.state_dict() 는 param 을 인덱스로만 참조한다 -> 등록 순서가 바뀐 코드에서
                # 재개해도 맞출 수 있게 이름 순서를 같이 남긴다 (위 resume 블록 참고).
                'opt_param_names': _opt_param_names,
                'global_step': global_step,
                'epoch': epoch,
                'best_val': best_val,
                'exp_name': exp_name,
                'wandb_id': _wandb_id,
            }, os.path.join(ckpt_dir, "resume.pth"))
        model.train()

    for epoch in range(start_epoch, cfg.epochs):
        pbar = tqdm(train_dataloader)
        model.train()
        total_loss = torch.tensor(0.0, device=accelerator.device)
        total_samples = torch.tensor(0, device=accelerator.device)
        t0 = time.time()
        for step, data in enumerate(pbar):
            t1 = time.time()
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(accelerator.device)
            B = traj.shape[0]

            t2 = time.time()
            with torch.no_grad():
                if cfg.load_points:
                    if 'pc' in data:
                        point = data['pc']
                        if cfg.point_encoder != 'custom':
                            pc_embeds, pc_masks, point_obj = pc_encoder(point)
                    elif 'pc_embeds' in data:
                        pc_embeds = data['pc_embeds'].to(device)
                        pc_masks = data['pc_masks'].to(device)
                    t3 = time.time()
                    # target_dir = '/home/ckd248/data/SCVideo/camera_generation/dataset/pc_embeds_datadop'
                    # for name, embed, points_mask in zip(data_name, pc_embeds, pc_masks):
                    #     file_path = os.path.join(target_dir, name)
                    #     valid_mask = points_mask > 0
                    #     # print(data_name[0])
                    #     # print(embed[valid_mask].shape)
                    #     # print(sim[valid_mask].shape)
                    #     # print(p2v_map.shape)
                    #     # print(inv.shape)
                    #     # print(points_single.shape)
                    #     # print(scatter_mean(points[0], inv, dim=0).shape)
                    #     # torch.save(torch.cat([points_single.cpu(), torch.ones(points_single.shape[0], 3)], dim=1), 'example_scene.pt')
                    #     # input()
                    #     # print(p2v_map[inv].shape)
                    #     # print(points_mask.shape)
                    #     # print(points_mask.sum())
                    #     # print(points_single.shape)
                    #     # print(p2v_map[inv].max())
                    #     # print(embed[valid_mask].shape)
                    #     # input()
                    #     # p2v_map = p2v_map[inv]
                    #     # voxel_coord = scatter_mean(points_single, p2v_map, dim=0) # (V, 3)
                    #     # print(voxel_coord.shape)
                    #     embed_combined = embed[valid_mask] # (V, D)
                    #     # embed_combined = torch.cat([voxel_coord, embed_combined], dim=-1)
                    #     embed_data = {
                    #         "embed": embed_combined.cpu(),
                    #         # "sim": sim[valid_mask].cpu(),
                    #         # "p2v_map": p2v_map.cpu(),
                    #         "mask": points_mask.cpu()
                    #     }
                    #     torch.save(embed_data, file_path)
                    # continue
                else:
                    pc_embeds, pc_masks = None, None
                if cfg.text_encoder == 'T5':
                    text_embeds, text_masks = text_encoder(text_prompt, device)
                elif cfg.text_encoder == 'CLIP':
                    text_embeds, text_masks = encode_text_clip(text_encoder, tokenizer, text_prompt, device=device)
                elif cfg.text_encoder == 'PEAV':
                    # [new 2026-09-03] arm B: umt5 대신 PE-AV text tower (1024-d) 를 text CA 로.
                    # 인코더를 여기서 돌리지 않고 데이터셋 캐시를 그대로 쓴다 (frozen + 캡션이
                    # 세그먼트 상수라 매 스텝 forward 할 이유가 없다). model 의 text_dim 도
                    # 1024 여야 한다 — 아래 모델 생성부가 cfg.text_encoder 를 보고 맞춘다.
                    text_embeds, text_masks = data['peav_text'].to(device), data['peav_text_mask'].to(device)
                text_embeds = text_embeds.float()
                text_masks = text_masks.bool()
                # [new 2026-09-03] video CA kwargs. 캐시가 없으면 빈 dict = 기존 호출.
                _vkw = build_video_cond(data, device)
                if cfg.use_vae:
                    traj_latents = camera_vae.encode(traj) / cfg.vae_latent_scale
                else:
                    traj_latents = traj
                t4 = time.time()
            
            if 'pc' in data and cfg.point_encoder == 'custom':
                pc_embeds, pc_masks, point_obj = model.pc_encoder(point)

            # Geo latent conditioning (image-based, frozen, on-the-fly). Replaces the
            # point-cloud path: feeds the latent model's geo_emb/geo_mask (5th/6th args).
            # 'geo_emb' present = the dataset served a precomputed cache hit (cfg.
            # geo_latent_cache_dir); otherwise fall back to the original LagerNVS forward.
            # [new 2026-08-07] geo_encoder='custom' 은 GeoTokenizer/proj 가 학습 대상이라
            # no_grad 로 감싸면 gradient 가 끊긴다. lagernvs 는 frozen 이므로 기존대로 no_grad.
            if 'geo_raw' in data:
                # pre-ln 캐시: frozen 구간만 건너뛰고 ln/proj 는 gradient 를 받아야 하므로
                # no_grad 로 감싸지 않는다 (얼린 쪽이 이미 backbone 안에서 no_grad 다).
                with accelerator.autocast():
                    pc_embeds, pc_masks = geo_emb_from_raw_cache(geo_encoder, data, device)
            elif 'geo_emb' in data:
                pc_embeds, pc_masks = geo_emb_from_cache(data, device)
            elif geo_encoder is not None and 'images' in data:
                if _geo_trainable:
                    # accelerator.prepare 를 태우지 않았으므로(우리는 forward 가 아니라
                    # forward_batch 로 부른다) autocast 를 여기서 직접 건다. 안 걸면 DINOv2
                    # ViT-L forward 만 fp32 로 돌아 mixed_precision='bf16' 설정이 무의미해진다.
                    with accelerator.autocast():
                        pc_embeds, pc_masks = geo_encode(geo_encoder, data, device)
                else:
                    with torch.no_grad():
                        pc_embeds, pc_masks = geo_encode(geo_encoder, data, device)
            if pc_embeds is not None:
                pc_embeds = attach_geo_cam(pc_embeds, data, device)

            noise = torch.randn_like(traj_latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (traj_latents.shape[0],),
                device=device,
            ).long()
            if getattr(cfg, 'is_ar', False):
                # chunk-wise AR: teacher-forced causal self-attn over past clean latents
                _raw = accelerator.unwrap_model(model)
                loss = ar_train_loss(_raw, noise_scheduler, traj_latents, text_embeds, text_masks,
                                     pc_embeds, pc_masks, cfg.ar_chunk_size)
            elif getattr(cfg, 'per_token_noise', False):
                # Diffusion Forcing: per-token tau rectified flow on the causal-VAE latent
                loss = per_token_flow_loss(model, traj_latents, text_embeds, text_masks,
                                           pc_embeds, pc_masks)
            else:
                noisy_x = noise_scheduler.add_noise(traj_latents, noise, timesteps)
                # target_track_dim=0 이면 _cond=None -> 기존 호출과 동일.
                _cond = build_track_cond(data, traj_latents.shape[1], device,
                                         dropout_p=float(getattr(cfg, 'target_track_dropout', 0.1)))
                noise_pred = model(noisy_x, timesteps.float(), text_embeds, text_masks,
                                   pc_embeds, pc_masks, cond=_cond, **_vkw)
                loss = F.mse_loss(noise_pred, noise)
                
            t5 = time.time()
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad(set_to_none=True)
            global_step += 1
            t6 = time.time()
            accelerator.log({"train/loss": loss.item()}, step=global_step)
            pbar.set_description(f"Epoch {epoch} | Loss {loss.item():.4f}")

            total_loss += loss.detach() * B
            total_samples += B
            # print(
            #     f"data {t1-t0:.3f} | "
            #     f"ray_sim {t2-t1:.3f} | "
            #     f"point {t3-t2:.3f} | "
            #     f"text {t4-t3:.3f} | "
            #     f"model {t5-t4:.3f} | "
            #     f"loss {t6-t5:.3f} | "
            #     f"total {t6-t0:.3f} | "
            # )
            t0 = time.time()

        total_loss = accelerator.reduce(total_loss, reduction='sum')
        total_samples = accelerator.reduce(total_samples, reduction='sum')
        accelerator.log(
            {"train/epoch_loss": total_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch": epoch + 1},
            step=global_step
        )

        # Validation once per epoch (wandb curve). run_validation restores train mode
        # and keeps only last.pth + best.pth (by val/loss_traj).
        run_validation(epoch, global_step)

if __name__ == "__main__":
    train()