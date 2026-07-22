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
    def __init__(self, backend="lagernvs", out_dim=768, repo_path=None, ckpt_path=None):
        super().__init__()
        self.backend_name = backend
        self.out_dim = out_dim

        if backend == "lagernvs":
            self.backend = _LagerNVSBackend(repo_path=repo_path, ckpt_path=ckpt_path)
        elif backend == "scenetok":
            self.backend = _SceneTokBackend(repo_path=repo_path, ckpt_path=ckpt_path)
        else:
            raise ValueError(f"unknown geo_encoder backend: {backend}")

        native = self.backend.out_dim
        # align to geo_latent_dim; Identity when they already match (lagernvs -> 768)
        self.proj = nn.Identity() if native == out_dim else nn.Linear(native, out_dim)
        if native != out_dim:
            print(f"(geo_encoder) proj {native} -> {out_dim} (trainable)")

    def forward(self, images, cam_token=None):
        """-> geo_embeds (B, M, out_dim), geo_masks (B, M) bool"""
        tokens, mask = self.backend(images, cam_token)   # backend is frozen / no_grad
        return self.proj(tokens), mask

    def build_cam_token(self, geo_c2w, geo_fxfycxcy, geo_hw, override_scale=None):
        """Delegate to the backend (posed geo). -> cam_token (B, V, 11)."""
        return self.backend.build_cam_token(geo_c2w, geo_fxfycxcy, geo_hw, override_scale=override_scale)


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
    raise ValueError(f"unknown geo_encoder backend: {backend}")
