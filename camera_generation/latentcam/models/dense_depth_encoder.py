"""이 코드가 답하는 질문: 소스 depth·카메라를 Molmo2 SigLIP 첫 입력만큼 dense 하게 (프레임당 378² 픽셀) 모델 조건으로 넣으려면?

R588 (사용자 10-28 "첫 입력을 Molmo2 처럼 dense 하게 · source camera 는 plucker · conv 로 downsample").
입력 (프레임마다, Molmo2 와 같은 늘린 S×S 좌표 — crop·pad 없음):
  depth  (B,3,T,S,S) = [log1p(z / norm_scale)·valid, valid, dynamic]  (dataset `_depth_vid_dense`, 캐시 fit/bank/depth_vid_cache.py --mode dense)
  Plücker (B,6,T,S,S) = 픽셀마다 ray (방향 d, moment o×d) — frame-s 카메라 좌표 / norm_scale. 늘린 픽셀을 원본 픽셀로 되돌려 원래 K 로 계산하므로
         가로가 눌린 격자여도 3D 는 정확하다. dataset 은 프레임별 (Rt 9, o 3, fx/W fy/H cx/W cy/H 4) = 16 만 주고 여기서 GPU 로 만든다.
학습 conv: Conv2d (patch 14, stride 14) = SigLIP 패치 격자 27×27 → 4 프레임씩 latent 시간칸 13 개로 평균 (토큰 τ = 프레임 4τ-3..4τ, τ=0 은 프레임 0)
  → 칸마다 학습 query Q 개 resampler (729 패치 토큰 + (y,x) Fourier PE) → (B, 13·Q, H). 시간 PE 는 호출 쪽 (카메라 토큰과 같은 축) 이 붙인다.
"""
import math

import torch
import torch.nn as nn


class _RS(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, kv):
        kvn = self.ln_kv(kv)
        q = q + self.attn(self.ln_q(q), kvn, kvn, need_weights=False)[0]
        return q + self.mlp(self.ln_m(q))


class DenseDepthEncoder(nn.Module):
    def __init__(self, hidden, heads, use_depth=True, use_plucker=True, patch=14, queries=16, layers=2, t_latent=13, stem_dim=128, nf=6):
        super().__init__()
        assert use_depth or use_plucker
        self.use_depth, self.use_plucker, self.patch, self.Q, self.TL, self.nf = use_depth, use_plucker, patch, queries, t_latent, nf
        c_in = (3 if use_depth else 0) + (6 if use_plucker else 0)
        self.stem = nn.Sequential(nn.Conv2d(c_in, stem_dim, patch, patch), nn.SiLU(), nn.Conv2d(stem_dim, stem_dim, 1))
        self.tok = nn.Linear(stem_dim, hidden)
        self.pe = nn.Sequential(nn.Linear(2 + 2 * 2 * nf, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.q = nn.Parameter(torch.randn(queries, hidden) * 0.02)
        self.rs = nn.ModuleList([_RS(hidden, heads) for _ in range(layers)])
        self.out = nn.LayerNorm(hidden)

    def plucker(self, cam, S, dtype):
        """cam (B,T,16) = Rt (9, c2w 회전) · o (3, 카메라 중심) · fx/W, fy/H, cx/W, cy/H → (B,6,T,S,S)."""
        B, T, _ = cam.shape; dev = cam.device
        Rt, o, k = cam[..., :9].reshape(B, T, 3, 3).float(), cam[..., 9:12].float(), cam[..., 12:16].float()
        g = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) / S                     # 늘린 격자 → 원본 정규화 좌표 (u = x/W)
        vv, uu = torch.meshgrid(g, g, indexing="ij")
        dx = (uu[None, None] - k[..., 2, None, None]) / k[..., 0, None, None]                 # (B,T,S,S)
        dy = (vv[None, None] - k[..., 3, None, None]) / k[..., 1, None, None]
        dc = torch.stack([dx, dy, torch.ones_like(dx)], -1)                                    # (B,T,S,S,3)
        d = torch.einsum("btij,btyxj->btyxi", Rt, dc)
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        m = torch.cross(o[:, :, None, None, :].expand_as(d), d, dim=-1)
        return torch.cat([d, m], -1).permute(0, 4, 1, 2, 3).to(dtype)                        # (B,6,T,S,S)

    def forward(self, depth=None, cam=None):
        """depth (B,3,T,S,S) · cam (B,T,16) → tokens (B, TL·Q, H), 시간칸 번호 (TL·Q,) long."""
        ref = depth if depth is not None else cam
        B = ref.shape[0]
        parts = []
        if self.use_depth:
            parts.append(depth)
        if self.use_plucker:
            S = depth.shape[-1] if depth is not None else 378
            parts.append(self.plucker(cam, S, (depth if depth is not None else cam).dtype))
        x = torch.cat(parts, 1)                                                                # (B,C,T,S,S)
        _, C, T, S, _ = x.shape
        f = self.stem(x.permute(0, 2, 1, 3, 4).reshape(B * T, C, S, S))                       # (B·T,D,n,n)
        n = f.shape[-1]; D = f.shape[1]
        f = f.reshape(B, T, D, n * n)
        # 프레임 → latent 시간칸 (causal VAE: 토큰 0 = 프레임 0, 토큰 τ = 프레임 4τ-3..4τ) 평균
        slot = torch.clamp((torch.arange(T, device=x.device) + 3) // 4, max=self.TL - 1)
        W = torch.zeros(self.TL, T, device=x.device, dtype=f.dtype); W[slot, torch.arange(T, device=x.device)] = 1
        W = W / W.sum(1, keepdim=True).clamp(min=1)
        f = torch.einsum("st,btdp->bsdp", W, f)                                               # (B,TL,D,P)
        kv = self.tok(f.permute(0, 1, 3, 2))                                                   # (B,TL,P,H)
        c1 = ((torch.arange(n, device=x.device, dtype=kv.dtype) + 0.5) / n) * 2 - 1
        yx = torch.stack(torch.meshgrid(c1, c1, indexing="ij"), -1).reshape(-1, 2)
        ang = yx[..., None] * (2.0 ** torch.arange(self.nf, device=x.device, dtype=kv.dtype)) * math.pi
        kv = kv + self.pe(torch.cat([yx, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1))[None, None]
        kv = kv.reshape(B * self.TL, n * n, -1)
        q = self.q.to(kv.dtype)[None].expand(B * self.TL, -1, -1)
        for blk in self.rs:
            q = blk(q, kv)
        tokens = self.out(q).reshape(B, self.TL * self.Q, -1)
        return tokens, torch.arange(self.TL, device=x.device).repeat_interleave(self.Q)
