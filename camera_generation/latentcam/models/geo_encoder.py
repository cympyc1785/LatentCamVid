"""Image-based scene *geo encoder* — mirror of pc_encoder.py, but instead of a
point-cloud (Concerto) encoder it wraps a pluggable multi-view scene encoder
(lagernvs / scenetok) and produces the geometry latent consumed by
camera_diffusion_model_latent.CameraDiffusionModel (its geo_emb / geo_mask
cross-attention).

Design (see plan): the backend encoder is FROZEN and run on-the-fly in the
training loop. GeoEncoder.forward(images, cam_token) -> (geo_embeds (B, M, out_dim),
geo_masks (B, M)), where out_dim == cfg.geo_latent_dim (768).

Backends:
  - 'lagernvs'  : LagerNVS Reconstructor (VGGT encoder + geo_feature_connector),
                  native token dim == renderer hidden (768) -> proj is Identity.
  - 'scenetok'  : SceneTok scene-token encoder (stub; wire when needed).
"""
import os
import sys
import contextlib

import torch
import torch.nn as nn
import einops


# --------------------------------------------------------------------------- #
# Backend: LagerNVS (VGGT reconstructor)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _shadow_models_package(repo_path):
    """Import a foreign repo that also uses a top-level `models` package without
    clobbering latentcam's own `models` package.

    latentcam and lagernvs both expose `models`; this temporarily removes
    latentcam's `models*` from sys.modules, puts `repo_path` first on sys.path so
    lagernvs's `models` resolves, then restores everything on exit.
    NOTE: validate this path once real data/env is available (see plan blockers).
    """
    saved = {k: v for k, v in list(sys.modules.items())
             if k == "models" or k.startswith("models.")}
    for k in list(saved):
        del sys.modules[k]
    sys.path.insert(0, repo_path)
    try:
        yield
    finally:
        if repo_path in sys.path:
            sys.path.remove(repo_path)
        # drop lagernvs's models.* and restore latentcam's
        for k in [k for k in list(sys.modules) if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.modules.update(saved)


class _LagerNVSBackend(nn.Module):
    """Frozen LagerNVS Reconstructor -> per-view scene tokens (B, V, P, C)."""

    def __init__(self, repo_path, ckpt_path,
                 attention_to_features_type="bidirectional_cross_attention"):
        super().__init__()
        if repo_path is None or not os.path.isdir(repo_path):
            raise FileNotFoundError(f"lagernvs_repo_path not found: {repo_path}")

        self.repo_path = repo_path
        with _shadow_models_package(repo_path):
            from models.encoder_decoder import EncDec_VitB8  # lagernvs's models
            from vggt.utils.pose_enc import extri_intri_to_pose_encoding
            self._pose_enc_fn = staticmethod(extri_intri_to_pose_encoding).__func__
            net = EncDec_VitB8(
                freeze_vggt=True,
                pretrained_vggt=False,   # weights come from ckpt
                attention_to_features_type=attention_to_features_type,
            )
            if ckpt_path and os.path.isfile(ckpt_path):
                sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                sd = sd.get("model", sd)
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
                missing, unexpected = net.load_state_dict(sd, strict=False)
                print(f"(geo_encoder/lagernvs) loaded {ckpt_path} "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
            else:
                print(f"(geo_encoder/lagernvs) WARNING: ckpt not found ({ckpt_path}); "
                      f"using randomly-initialized reconstructor")

        # keep only the encoder (reconstructor); drop the renderer
        self.reconstructor = net.reconstructor
        for p in self.reconstructor.parameters():
            p.requires_grad_(False)
        self.reconstructor.eval()

        # renderer hidden size == token channel dim (EncDecVitB/8 -> 768)
        self.out_dim = self.reconstructor.geo_feature_connector.out_features
        self.camera_encoding_dim = getattr(self.reconstructor, "camera_encoding_dim", 11)

    @torch.no_grad()
    def build_cam_token(self, geo_c2w, geo_fxfycxcy, geo_hw, override_scale=None):
        """Build the 11-dim lagernvs cam_token (posed) from geo-view geometry, matching
        data.normalization.build_cam_cond: normalize c2w relative to the first view and
        divide translations by 1.35*max(||cond center||), encode via VGGT's
        extri_intri_to_pose_encoding (absT_quaR_FoV, 9-dim), append [camera_scale, 0].

        geo_c2w: (B, V, 4, 4) OpenCV c2w; geo_fxfycxcy: (B, V, 4) px; geo_hw: (B, 2) [h, w].
        override_scale: (B,) or (B,1) — if given, use this as the context scene_scale instead
        of the internal 1.35*max(||cond center||). Pass the target's initial 1.35*max_dist scale
        so context & target share ONE frame+scale (skips LagerNVS's own context normalization).
        returns cam_token: (B, V, 11).
        """
        B, V = geo_c2w.shape[:2]
        c2w = geo_c2w.float().clone()
        first_inv = torch.linalg.inv(c2w[:, 0:1])                      # (B,1,4,4)
        c2w = first_inv @ c2w                                          # relative to view 0
        if override_scale is not None:                                 # use target's 1.35*max_dist
            scene_scale = override_scale.float().reshape(B).clamp(min=1e-6).view(B, 1, 1)
        else:                                                          # LagerNVS's own 1.35*ctx_max
            scene_scale = 1.35 * torch.linalg.norm(c2w[:, :, :3, 3], dim=-1).amax(dim=1)  # (B,)
            scene_scale = scene_scale.clamp(min=1e-6).view(B, 1, 1)
        c2w[:, :, :3, 3] = c2w[:, :, :3, 3] / scene_scale
        camera_scale = torch.linalg.norm(c2w[:, :, :3, 3], dim=-1).amax(dim=1)  # (B,)
        # per-sample FoV needs a single hw; DL3DV shares intrinsics/resolution per scene
        toks = []
        for b in range(B):
            hw = (int(geo_hw[b, 0].item()), int(geo_hw[b, 1].item()))
            pose9 = self._pose_enc_fn(c2w[b:b + 1], geo_fxfycxcy[b:b + 1].float(),
                                      image_size_hw=hw)                # (1,V,9)
            toks.append(pose9[0])
        pose9 = torch.stack(toks, 0)                                  # (B,V,9)
        scale_tok = torch.zeros(B, V, 2, device=pose9.device, dtype=pose9.dtype)
        scale_tok[:, :, 0] = camera_scale.view(B, 1)
        return torch.cat([pose9, scale_tok], dim=-1)                  # (B,V,11)

    @torch.no_grad()
    def forward(self, images, cam_token=None):
        """images: (B, V, 3, H, W); cam_token: (B, V, cam_dim) or None (-> zeros).
        returns tokens (B, M, out_dim), mask (B, M) bool (all valid)."""
        b, v = images.shape[0], images.shape[1]
        if cam_token is None:
            cam_token = images.new_zeros(b, v, self.camera_encoding_dim)
        tokens = self.reconstructor(images, cam_token)          # (B, V, P, C)
        tokens = einops.rearrange(tokens, "b v p c -> b (v p) c")  # (B, M, C)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


class _CustomBackend(nn.Module):
    """custom_geo_encoder.SceneEncoder 래퍼 — RGB + Plücker + depth 를 받는 자체 설계 encoder.

    lagernvs 와 두 가지가 다르다:
      1) TRAINABLE (frozen 은 안쪽 DINOv2 뿐) -> optimizer / no_grad / ckpt 저장이 달라진다.
      2) forward 인자가 images 하나가 아니라 batch dict (Plücker/depth/valid 가 더 필요) ->
         wants_batch=True 로 표시하고 GeoEncoder.forward_batch 로 부른다.
    """

    trainable = True
    wants_batch = True

    def __init__(self, cfg):
        super().__init__()
        from models.custom_geo_encoder import SceneEncoder
        self.net = SceneEncoder(
            out_dim=int(getattr(cfg, 'geo_latent_dim', 768)),
            dino_path=getattr(cfg, 'custom_geo_dino_path', None),
            freeze_dino=bool(getattr(cfg, 'custom_geo_freeze_dino', True)),
            input_hw=getattr(cfg, 'custom_geo_input_hw', None),
            c_ray=int(getattr(cfg, 'custom_geo_ray_dim', 64)),
            c_geo=int(getattr(cfg, 'custom_geo_geo_dim', 256)),
            channels=getattr(cfg, 'custom_geo_channels', 'full'),
        )
        self.out_dim = self.net.out_dim
        self.camera_encoding_dim = 0     # 카메라는 cam_token 이 아니라 픽셀 Plücker 로 들어간다
        # [ablation] 배치에서 실제로 필요한 키. train 쪽이 이걸 보고 .to(device) 할 것만 옮긴다.
        self.needs_keys = {
            'full': ('images', 'geo_plucker_map', 'geo_logd', 'geo_valid'),
            'no_depth': ('images', 'geo_plucker_map'),
            'rgb_only': ('images',),
        }[self.net.channels]

    def forward(self, batch):
        """batch: 'images'/'geo_plucker_map'/'geo_logd'/'geo_valid' -> (B, M, C), (B, M) bool"""
        tokens = self.net(batch['images'], batch.get('geo_plucker_map'),
                          batch.get('geo_logd'), batch.get('geo_valid'))
        # mask 는 일단 전부 유효로 둔다 (depth 없는 view 를 context 에 섞지 않는다는 전제).
        # 섞게 되면 valid 비율이 낮은 patch 를 여기서 False 로 내려야 한다.
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


class _DA3Backend(nn.Module):
    """da3_geo_encoder.DA3SceneEncoder 래퍼 — Depth-Anything-3 의 cross-view ViT 백본.

    표면은 `_LagerNVSBackend` 와 **동일하게** 맞춰 뒀다 (`wants_batch=False`,
    `build_cam_token(...)`, `forward(images, cam_token)`). 그래서 `geo_encode` 의 5개 사본
    (train_latent_cam_dm / infer_validation_sample / infer_swap_ablation / eval_testset /
    eval/geo_ablation) 을 한 줄도 고치지 않아도 된다.

    다만 `trainable = True` 다. DA3 본체는 frozen 이지만 그 위의 LayerNorm 과
    GeoEncoder.proj(3072->768) 는 학습돼야 하고, 이 플래그가 꺼지면
      (1) 두 모듈이 optimizer group 에 안 들어가 랜덤 초기값에 영원히 머물고
      (2) frozen-key ckpt 필터와 accelerator.autocast() 도 안 걸린다.
    frozen 파라미터는 `_geo_frozen_keys` 가 ckpt 에서 빼므로 last_geo.pth 는 여전히 작다.
    """

    trainable = True
    wants_batch = False

    def __init__(self, cfg):
        super().__init__()
        from models.da3_geo_encoder import DA3SceneEncoder
        self.net = DA3SceneEncoder(
            repo_path=getattr(cfg, 'da3_geo_repo_path', None),
            model_name=str(getattr(cfg, 'da3_geo_model', 'da3nested-giant-large')),
            ckpt_path=getattr(cfg, 'da3_geo_ckpt_path', None),
            hf_home=getattr(cfg, 'da3_geo_hf_home', None),
            input_hw=getattr(cfg, 'da3_geo_input_hw', None),
            layers=getattr(cfg, 'da3_geo_layers', 'last'),
            layer_fuse=getattr(cfg, 'da3_geo_layer_fuse', 'concat'),
            norm=getattr(cfg, 'da3_geo_norm', 'ln'),
            debug=bool(getattr(cfg, 'da3_geo_debug', False)),
            cam_token_per_sample=bool(getattr(cfg, 'da3_cam_token_per_sample', False)),
        )
        self.out_dim = self.net.out_dim
        self.camera_encoding_dim = self.net.camera_encoding_dim

    def build_cam_token(self, geo_c2w, geo_fxfycxcy, geo_hw, override_scale=None):
        return self.net.build_cam_token(geo_c2w, geo_fxfycxcy, geo_hw,
                                        override_scale=override_scale)

    def forward(self, images, cam_token=None):
        """cam_token=None 이면 DA3 가 자기 learned camera_token 을 쓴다 (unposed).
        lagernvs 처럼 zeros 를 넣으면 안 된다 — 학습된 토큰 자리를 0 으로 덮게 된다."""
        return self.from_raw(self.encode_raw(images, cam_token))

    # [new 2026-08-29] pre-ln 캐시 진입점. 자세한 근거는 da3_geo_encoder.encode_raw docstring.
    def encode_raw(self, images, cam_token=None):
        """frozen backbone 구간만 -> (B,V,P,C_native) pre-ln. 캐시 빌더가 쓴다."""
        return self.net.encode_raw(images, cam_token)

    def from_raw(self, t):
        """pre-ln 토큰(또는 그 캐시) -> (tokens (B,V*P,C), mask). ln 은 여기서 학습된다."""
        tokens = self.net.from_raw(t)                        # (B, V*P, C)
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


class _SceneTokBackend(nn.Module):
    """SceneTok scene-token encoder (stub). Wire when needed:
    load the SceneTok encoder, run on multi-view input -> scene tokens (B, M, D),
    set self.out_dim = D, return (tokens, mask)."""

    def __init__(self, repo_path, ckpt_path):
        super().__init__()
        raise NotImplementedError(
            "scenetok geo backend not implemented yet — plan implements lagernvs first")


# --------------------------------------------------------------------------- #
# GeoEncoder wrapper (pc_encoder-style interface)
# --------------------------------------------------------------------------- #
class GeoEncoder(nn.Module):
    def __init__(self, backend="lagernvs", out_dim=768, repo_path=None, ckpt_path=None,
                 cfg=None):
        super().__init__()
        self.backend_name = backend
        self.out_dim = out_dim

        if backend == "lagernvs":
            self.backend = _LagerNVSBackend(repo_path=repo_path, ckpt_path=ckpt_path)
        elif backend == "scenetok":
            self.backend = _SceneTokBackend(repo_path=repo_path, ckpt_path=ckpt_path)
        elif backend == "custom":
            self.backend = _CustomBackend(cfg)
        elif backend == "da3":
            self.backend = _DA3Backend(cfg)
        else:
            raise ValueError(f"unknown geo_encoder backend: {backend}")

        # lagernvs/scenetok 은 frozen + images 인자, custom 은 trainable + batch dict 인자
        self.trainable = bool(getattr(self.backend, "trainable", False))
        self.wants_batch = bool(getattr(self.backend, "wants_batch", False))
        self.needs_keys = tuple(getattr(self.backend, "needs_keys",
                                        ('images', 'geo_plucker_map', 'geo_logd', 'geo_valid')))

        native = self.backend.out_dim
        # align to geo_latent_dim; Identity when they already match (lagernvs -> 768)
        self.proj = nn.Identity() if native == out_dim else nn.Linear(native, out_dim)
        if native != out_dim:
            print(f"(geo_encoder) proj {native} -> {out_dim} (trainable)")

    def forward(self, images, cam_token=None):
        """-> geo_embeds (B, M, out_dim), geo_masks (B, M) bool"""
        if self.wants_batch:
            raise TypeError(f"geo backend '{self.backend_name}' 는 batch dict 이 필요하다 "
                            f"-> forward_batch(batch) 를 써라")
        tokens, mask = self.backend(images, cam_token)   # backend is frozen / no_grad
        return self.proj(tokens), mask

    def forward_batch(self, batch):
        """batch dict 을 통째로 받는 backend(custom)용 진입점. lagernvs 는 images/cam_token 만
        쓰므로 이 경로로 들어와도 기존과 동일하게 동작한다 (cam_token 은 호출자가 붙여 넘길 것)."""
        if not self.wants_batch:
            return self(batch['images'], batch.get('cam_token'))
        tokens, mask = self.backend(batch)
        return self.proj(tokens), mask

    def build_cam_token(self, geo_c2w, geo_fxfycxcy, geo_hw, override_scale=None):
        """Delegate to the backend (posed geo). -> cam_token (B, V, 11)."""
        return self.backend.build_cam_token(geo_c2w, geo_fxfycxcy, geo_hw, override_scale=override_scale)

    # ------------------------------------------------------------------ #
    # [new 2026-08-29] pre-ln 캐시 (cfg.geo_raw_cache_dir).
    # `geo_latent_cache_dir` 은 **proj 뒤**를 얼리므로 backend 가 trainable 이면 못 쓴다.
    # 이 쌍은 그 선을 backend 안쪽(ln 직전)으로 옮긴 것이라 ln/proj 는 그대로 학습된다.
    # da3 backend 만 지원한다 (lagernvs 는 애초에 frozen 이라 기존 캐시로 충분).
    # ------------------------------------------------------------------ #
    def encode_raw(self, images, cam_token=None):
        """frozen backend 구간만 -> pre-ln 토큰 (B, V, P, C_native). 캐시 빌더 전용."""
        if not hasattr(self.backend, 'encode_raw'):
            raise TypeError(f"geo backend '{self.backend_name}' 는 pre-ln 캐시를 지원하지 않는다 "
                            f"(da3 만 encode_raw/from_raw 를 가진다)")
        return self.backend.encode_raw(images, cam_token)

    def from_raw(self, t):
        """pre-ln 토큰(또는 그 캐시) -> geo_embeds (B, M, out_dim), geo_masks (B, M) bool."""
        if not hasattr(self.backend, 'from_raw'):
            raise TypeError(f"geo backend '{self.backend_name}' 는 pre-ln 캐시를 지원하지 않는다")
        tokens, mask = self.backend.from_raw(t)
        return self.proj(tokens), mask


def build_geo_encoder(cfg):
    """Factory reading cfg (mirror of the point_encoder selection in the train script)."""
    backend = getattr(cfg, "geo_encoder", "lagernvs")
    out_dim = getattr(cfg, "geo_latent_dim", 768)
    if backend == "lagernvs":
        return GeoEncoder(backend, out_dim,
                          repo_path=cfg.lagernvs_repo_path,
                          ckpt_path=cfg.lagernvs_ckpt_path)
    elif backend == "scenetok":
        return GeoEncoder(backend, out_dim,
                          repo_path=getattr(cfg, "scenetok_repo_path", None),
                          ckpt_path=getattr(cfg, "scenetok_ckpt_path", None))
    elif backend == "custom":
        # dino 경로는 cfg 에 없으면 checkpoints/dinov2-large 로 떨어뜨린다 (hydra_cfg 가
        # ckpt_root 를 런타임에 채운다). cfg 를 통째로 넘기는 유일한 backend.
        if not getattr(cfg, "custom_geo_dino_path", None):
            root = getattr(cfg, "ckpt_root", None)
            if root:
                cfg.custom_geo_dino_path = os.path.join(root, "dinov2-large")
        return GeoEncoder(backend, out_dim, cfg=cfg)
    elif backend == "da3":
        # da3 도 cfg 통째로 (da3_geo_* 키가 여럿이다). 경로 기본값은 config.py 가 준다.
        return GeoEncoder(backend, out_dim, cfg=cfg)
    raise ValueError(f"unknown geo_encoder backend: {backend}")
