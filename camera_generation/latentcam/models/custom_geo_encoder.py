"""Custom geo encoder — RGB + Plücker + depth 를 받아 geo 토큰을 만드는, 직접 설계한
scene context encoder. LagerNVS(models/geo_encoder.py:_LagerNVSBackend) 대체용이며
cfg.geo_encoder = 'custom' 일 때만 쓰인다 (lagernvs 경로는 그대로 유지).

두 갈래:
  DINOv2 (frozen)  : 외관/semantic.  ViT-L/14 -> patch 당 1024-d
  GeoTokenizer     : 기하.  8ch = Plücker(6) + log-depth(1) + valid(1) -> patch 당 C_geo

두 갈래를 patch 단위로 concat -> Linear -> (B, V*P, out_dim). LagerNVS 와 달리
카메라 정보가 cam_token(11-d, view 당 1개)이 아니라 **픽셀당 Plücker ray** 로 들어간다.

입력 계약 (dataset 쪽은 아직 미구현 — geo_rgbd 경로를 붙일 때 이 shape 로 내보낼 것):
  rgb     (B, V, 3, H, W)  float [0, 1]   — dataset_dl3dv._load_images 가 주는 그대로 (정규화 X)
  plucker (B, V, 6, H, W)  target segment 첫 카메라 프레임, translation 은 /norm_scale
                           (dataset_dl3dv._geo_cam_plucker 와 같은 규약, 단 patch 격자가 아니라 픽셀 격자)
  logd    (B, V, 1, H, W)  log(depth / norm_scale)  — plucker 와 같은 단위여야 ray×depth 가 의미를 가진다
  valid   (B, V, 1, H, W)  depth 유효 마스크 {0,1}

출력: tokens (B, V*P, out_dim). geo mask 는 backend 쪽에서 all-True 로 만든다
(현재는 depth 없는 view 를 섞지 않는다는 전제 — models/geo_encoder.py:_CustomBackend).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GeoTokenizer(nn.Module):
    """8ch = Plücker(6) + log-depth(1) + valid(1) -> (N, C_tot, H/patch, W/patch).

    ray  : 저주파(카메라 광선)라 그냥 linear patchify.
    stem : depth × ray 상호작용을 conv 로 만든 뒤 patch 격자로 내린다. patch 가 짝수라는
           전제로 stride 2 -> stride patch//2 로 쪼갠다 (patch=14 -> 2*7, 16 -> 2*8).
    """

    def __init__(self, patch=14, c_ray=64, c_tot=256, use_depth=True):
        super().__init__()
        if patch % 2:
            raise ValueError(f"GeoTokenizer: patch 는 짝수여야 한다 (got {patch})")
        if c_tot <= c_ray:
            raise ValueError(f"GeoTokenizer: c_tot({c_tot}) > c_ray({c_ray}) 여야 한다")
        self.patch = patch
        # [ablation] use_depth=False 면 stem 이 Plücker(6)만 받는다. 채널 수 말고는
        # 나머지 구조/폭이 전부 같아서 depth 유무만 바뀐다.
        self.use_depth = bool(use_depth)
        c_in = 8 if self.use_depth else 6
        self.ray = nn.Conv2d(6, c_ray, patch, patch)
        c_s = c_tot - c_ray
        self.stem = nn.Sequential(
            nn.Conv2d(c_in, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 96, 3, 2, 1), nn.GroupNorm(8, 96), nn.GELU(),      # /2
            nn.Conv2d(96, 128, 3, 1, 1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, c_s, patch // 2, patch // 2),                     # /patch 누적
        )
        self.out_dim = c_tot

    def forward(self, plucker, logd=None, valid=None):
        """plucker (N,6,H,W), logd (N,1,H,W), valid (N,1,H,W) -> (N, C_tot, H/patch, W/patch)"""
        if self.use_depth:
            logd = logd * valid
            x = torch.cat([plucker, logd, valid], 1)
        else:
            x = plucker
        return torch.cat([self.ray(plucker), self.stem(x)], 1)


class SceneEncoder(nn.Module):
    """frozen DINOv2 + trainable GeoTokenizer -> geo 토큰 (B, V*P, out_dim)."""

    CHANNELS = ('full', 'no_depth', 'rgb_only')

    def __init__(self, out_dim=768, dino_path=None, freeze_dino=True,
                 input_hw=None, c_ray=64, c_geo=256, channels='full'):
        super().__init__()
        from transformers import Dinov2Model               # 지연 import (text-only 경로 영향 X)

        # [ablation 2026-08-10] channels
        #   full     : 기존 그대로 (Plücker + log-depth + valid)
        #   no_depth : GeoTokenizer 는 남기되 depth/valid 채널만 뺀다 (Plücker 만)
        #   rgb_only : GeoTokenizer 자체를 제거 -> frozen DINOv2 특징만 proj
        channels = str(channels or 'full')
        if channels not in self.CHANNELS:
            raise ValueError(f"custom_geo_channels: {self.CHANNELS} 중 하나여야 한다 (got {channels})")
        self.channels = channels

        if not dino_path or not os.path.isdir(dino_path):
            raise FileNotFoundError(f"custom_geo_dino_path not found: {dino_path}")
        self.dino = Dinov2Model.from_pretrained(dino_path)
        self.patch = int(self.dino.config.patch_size)      # ViT-L/14 -> 14
        d_dim = int(self.dino.config.hidden_size)          # ViT-L    -> 1024
        self.freeze_dino = bool(freeze_dino)
        if self.freeze_dino:
            for p in self.dino.parameters():
                p.requires_grad_(False)
            self.dino.eval()

        # DINO 가 학습된 정규화 그대로 (preprocessor_config.json 이 있으면 거기서 읽는다)
        mean, std = _IMAGENET_MEAN, _IMAGENET_STD
        pp = os.path.join(dino_path, 'preprocessor_config.json')
        if os.path.isfile(pp):
            with open(pp) as f:
                _pp = json.load(f)
            mean = tuple(_pp.get('image_mean', mean))
            std = tuple(_pp.get('image_std', std))
        self.register_buffer('_mean', torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std', torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        self.input_hw = tuple(input_hw) if input_hw else None   # None -> 첫 forward 에서 확정
        self.ln_d = nn.LayerNorm(d_dim)
        if channels == 'rgb_only':
            self.geo = None
            self.ln_g = None
            self.proj = nn.Linear(d_dim, out_dim)
        else:
            self.geo = GeoTokenizer(self.patch, c_ray, c_geo,
                                    use_depth=(channels == 'full'))
            self.ln_g = nn.LayerNorm(c_geo)
            self.proj = nn.Linear(d_dim + c_geo, out_dim)       # zero-init 하지 않음
        self.out_dim = out_dim
        self._warned_resize = False

    # ------------------------------------------------------------------ #
    def _resolve_hw(self, h, w):
        """DINOv2 는 H, W 가 patch 의 배수여야 한다. cfg 로 안 주면 내림해서 맞춘다
        (geo_image_hw (256,448) -> (252,448) -> 격자 18x32 = 576 tok/view)."""
        if self.input_hw is None:
            p = self.patch
            self.input_hw = ((h // p) * p, (w // p) * p)
        return self.input_hw

    def patch_grid(self, h=None, w=None):
        """(Gh, Gw). dataset 쪽에서 픽셀 격자/토큰 정렬을 맞출 때 쓸 것."""
        hw = self._resolve_hw(h, w) if self.input_hw is None else self.input_hw
        return hw[0] // self.patch, hw[1] // self.patch

    def _to_input_hw(self, rgb, plucker, logd, valid):
        """(N,C,H,W) 들을 DINO 격자에 맞는 해상도로 맞춘다. 이미 맞으면 no-op.

        plucker 를 bilinear 로 줄이면 이웃 ray 의 선형결합이라 방향 단위벡터가 정확히
        보존되지는 않는다 (작은 축소에서는 무시할 수준). 정확히 맞추려면 dataset 이
        애초에 input_hw 격자로 Plücker 를 만들어 주는 게 맞다 — 아래 경고가 그 신호다.
        """
        th, tw = self._resolve_hw(rgb.shape[-2], rgb.shape[-1])
        if rgb.shape[-2:] == (th, tw):
            return rgb, plucker, logd, valid
        if not self._warned_resize:
            print(f"(custom_geo) resize {tuple(rgb.shape[-2:])} -> ({th}, {tw}) "
                  f"(patch {self.patch} 배수). dataset 이 이 해상도로 직접 주는 편이 정확하다.")
            self._warned_resize = True
        rgb = F.interpolate(rgb, (th, tw), mode='bicubic', align_corners=False, antialias=True)
        # ablation arm 에서는 안 쓰는 입력이 None 으로 들어온다 -> 건드리지 않는다
        if plucker is not None:
            plucker = F.interpolate(plucker, (th, tw), mode='bilinear', align_corners=False)
        if logd is not None:
            logd = F.interpolate(logd, (th, tw), mode='bilinear', align_corners=False)
        if valid is not None:
            valid = F.interpolate(valid, (th, tw), mode='nearest')
        return rgb.clamp(0, 1), plucker, logd, valid

    def forward(self, rgb, plucker=None, logd=None, valid=None):
        """rgb (B,V,3,H,W) in [0,1], plucker (B,V,6,H,W), logd/valid (B,V,1,H,W)
        -> tokens (B, V*P, out_dim). ablation arm 은 안 쓰는 인자가 None 이다."""
        b, v = rgb.shape[:2]
        flat = lambda x: x.flatten(0, 1) if x is not None else None
        rgb, plucker, logd, valid = self._to_input_hw(
            flat(rgb), flat(plucker), flat(logd),
            flat(valid.float()) if valid is not None else None)

        x = (rgb - self._mean) / self._std
        if self.freeze_dino:
            with torch.no_grad():
                d = self.dino(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        else:
            d = self.dino(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        d = d[:, 1:]                                   # CLS 제거 (dinov2-large 는 register 토큰 없음)

        if self.geo is None:                           # rgb_only: DINO 특징만
            s = self.proj(self.ln_d(d))                # (N, P, out_dim)
            return s.view(b, v * s.shape[1], self.out_dim)

        g = self.geo(plucker, logd, valid)             # (N, C_geo, Gh, Gw)
        gh, gw = g.shape[-2:]
        g = g.flatten(2).transpose(1, 2)               # (N, P, C_geo)
        if d.shape[1] != g.shape[1]:
            raise RuntimeError(f"token 수 불일치: dino {d.shape[1]} vs geo {g.shape[1]} "
                               f"(격자 {gh}x{gw}, patch {self.patch}, input_hw {self.input_hw})")

        s = self.proj(torch.cat([self.ln_d(d.to(g.dtype)), self.ln_g(g)], -1))   # (N, P, out_dim)
        return s.view(b, v * s.shape[1], self.out_dim)
