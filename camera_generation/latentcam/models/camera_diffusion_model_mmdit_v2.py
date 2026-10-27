"""[R580] MMDiT 판 카메라 latent diffusion — 사용자 10-27 "depth, source camera 를 MMDiT 참고해서 Molmo2 feature 랑 text 랑 같이".

WorldDirector `baseline_mmdit` (models/camera_diffusion_model_mmdit.py, 같은 lucidrains generalized MMDiT 블록) 의 확장:
  stream ① camera  : VAE latent T 토큰 (+ track·box 조건 concat, 기존 cond_dim 규약) → cam_in + 시간 PE
  stream ② text    : umt5 (B,L,4096) → proj + 1D 순서 PE (시간축 없음)
  stream ③ video   : Molmo2 (B,F·64,2560) → 프레임마다 학습 query 4 개로 attention pool → (B,F·4) + 시간 PE + query 임베딩.
                     Molmo2 문장 prefill (B,Lt,2560) 은 앞에 붙이되 1D 순서 PE + 종류 임베딩만 (시간 PE 없음)
  stream ④ geometry: depth 영상 (B,2,F,h,w) → Conv3d ×3 (stride 2) → 토큰 + 시간 PE + (y,x) Fourier PE
                     + 소스 카메라 (B,F,54) Plücker → MLP + 시간 PE, 종류 임베딩으로 구분
시간 PE 는 모든 프레임 유래 토큰이 같은 축 (소스 프레임 번호, 실수) 의 sinusoid 를 쓴다: camera 토큰 i → i·(F−1)/(T−1), depth 시간칸 중심, video·srccam 프레임 f.
블록 = MMDiTBlock (stream 별 AdaLN(timestep)·QKV·FF, joint attention, qk RMSNorm), 출력은 camera stream 에서만.
조건 결손: depth·srccam 은 학습 중 dropout 확률로 해당 stream 토큰을 mask (전부 빠지면 그 stream 은 한 개 null 토큰만 보인다).
forward 서명은 CameraDiffusionModel 과 같아 train/eval 스크립트가 그대로 부른다 (쓰지 않는 인자는 무시).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdit.mmdit_generalized_pytorch import MMDiT

from models.camera_diffusion_model_latent import (CrossAttention, TimeEmbedding, positional_encoding,
                                                  positional_encoding_at)


class CameraDiffusionMMDiT(nn.Module):
    def __init__(self, cam_dim=64, cond_dim=0, time_emb_dim=512, hidden_dim=512, num_layers=8, num_heads=8, text_dim=4096,
                 video_latent_dim=2560, video_text_dim=2560, video_queries=4, video_patches=64, geo_latent_dim=54,
                 depth_vid_cond=True, depth_vid_dropout=0.1, srccam_dropout=0.1, dropout=0.1, **_unused):
        super().__init__()
        self.cond_dim, self.hidden_dim, self.num_layers = int(cond_dim), hidden_dim, num_layers
        self.cam_in = nn.Linear(cam_dim + self.cond_dim, hidden_dim)
        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        # ② text
        self.text_ln, self.text_proj = nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim)
        # ③ video (Molmo2)
        self.video_latent_dim, self.video_text_dim = int(video_latent_dim), int(video_text_dim)
        self.vq, self.vp = int(video_queries), int(video_patches)
        if self.video_latent_dim > 0:
            self.video_ln, self.video_proj = nn.LayerNorm(self.video_latent_dim), nn.Linear(self.video_latent_dim, hidden_dim)
            self.video_q = nn.Parameter(torch.randn(self.vq, hidden_dim) * 0.02)
            self.video_pool = CrossAttention(hidden_dim, num_heads)
            self.video_pool_ln = nn.LayerNorm(hidden_dim)
        if self.video_text_dim > 0:
            self.vtext_ln, self.vtext_proj = nn.LayerNorm(self.video_text_dim), nn.Linear(self.video_text_dim, hidden_dim)
        self.video_type = nn.Parameter(torch.zeros(2, hidden_dim))          # [prefill 문장, 영상]
        # ④ geometry
        self.depth_vid_cond, self.dv_dropout, self.sc_dropout, self.dv_nf = bool(depth_vid_cond), float(depth_vid_dropout), float(srccam_dropout), 6
        if self.depth_vid_cond:
            self.dv_enc = nn.Sequential(nn.Conv3d(2, 32, 3, 2, 1), nn.SiLU(), nn.Conv3d(32, 64, 3, 2, 1), nn.SiLU(),
                                        nn.Conv3d(64, 128, 3, 2, 1), nn.SiLU())
            self.dv_tok = nn.Linear(128, hidden_dim)
            self.dv_xy_pe = nn.Sequential(nn.Linear(2 + 2 * 2 * self.dv_nf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.geo_latent_dim = int(geo_latent_dim or 0)
        if self.geo_latent_dim > 0:
            self.sc_in = nn.Sequential(nn.Linear(self.geo_latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.geo_type = nn.Parameter(torch.zeros(2, hidden_dim))            # [depth, srccam]
        self.geo_null = nn.Parameter(torch.zeros(1, 1, hidden_dim))        # 기하 조건이 전부 빠졌을 때 자리 (mask 전부 False 방지)
        self.layers = MMDiT(depth=num_layers, dim_modalities=(hidden_dim,) * 4, dim_cond=hidden_dim, qk_rmsnorm=True, num_residual_streams=1,
                            heads=num_heads)
        self.out = nn.Linear(hidden_dim, cam_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # train/eval 스크립트가 읽는 계측 속성 (DiT 판과 이름 맞춤)
        self.text_cross_attn_weight = self.geo_cross_attn_weight = self.video_cross_attn_weight = None
        self.readout = None; self.readout_aux_pred = None

    def _pe_at(self, pos, dtype):
        return positional_encoding_at(pos, self.hidden_dim).to(dtype)

    def forward(self, x_t, t, text_emb, text_mask, geo_emb=None, geo_mask=None, cond=None, video_emb=None, video_mask=None,
                video_text_emb=None, video_text_mask=None, depth_vid=None, **_unused):
        B, T, _ = x_t.shape
        dev, dt = x_t.device, x_t.dtype
        Fsrc = 49
        # ① camera
        if self.cond_dim > 0:
            c = cond if cond is not None else x_t.new_zeros(B, T, self.cond_dim)
            x_in = torch.cat([x_t, c.to(dt)], -1)
        else:
            x_in = x_t
        cam_pos = torch.arange(T, device=dev, dtype=torch.float32) * (Fsrc - 1) / max(T - 1, 1)
        h_cam = self.cam_in(x_in) + self._pe_at(cam_pos, dt)[None]
        # ② text (1D 순서 PE)
        h_txt = self.text_proj(self.text_ln(text_emb.to(dt)))
        h_txt = h_txt + positional_encoding(h_txt.shape[1], self.hidden_dim, device=dev).to(dt)[None]
        m_txt = text_mask.bool() if text_mask is not None else None
        # ③ video
        vparts, vmasks = [], []
        if self.video_text_dim > 0 and video_text_emb is not None:
            vt = self.vtext_proj(self.vtext_ln(video_text_emb.to(dt)))
            vt = vt + positional_encoding(vt.shape[1], self.hidden_dim, device=dev).to(dt)[None] + self.video_type[0]
            vparts.append(vt); vmasks.append(video_text_mask.bool() if video_text_mask is not None else torch.ones(B, vt.shape[1], dtype=torch.bool, device=dev))
        if self.video_latent_dim > 0 and video_emb is not None:
            Fv = video_emb.shape[1] // self.vp
            v = self.video_proj(self.video_ln(video_emb.to(dt))).reshape(B * Fv, self.vp, -1)
            q = self.video_q.to(dt)[None].expand(B * Fv, -1, -1)
            pooled = self.video_pool_ln(self.video_pool(q, v)).reshape(B, Fv * self.vq, -1)
            fpos = torch.arange(Fv, device=dev, dtype=torch.float32).repeat_interleave(self.vq)
            pooled = pooled + self._pe_at(fpos, dt)[None] + self.video_type[1]
            vparts.append(pooled)
            vm = video_mask.reshape(B, Fv, self.vp).any(-1).repeat_interleave(self.vq, 1) if video_mask is not None else torch.ones(B, Fv * self.vq, dtype=torch.bool, device=dev)
            vmasks.append(vm.bool())
        if not vparts:
            vparts, vmasks = [x_t.new_zeros(B, 1, self.hidden_dim)], [torch.ones(B, 1, dtype=torch.bool, device=dev)]
        h_vid, m_vid = torch.cat(vparts, 1), torch.cat(vmasks, 1)
        # ④ geometry (depth + srccam), 학습 중 stream 별 dropout = mask
        gparts, gmasks = [self.geo_null.to(dt).expand(B, -1, -1)], [torch.ones(B, 1, dtype=torch.bool, device=dev)]
        if self.depth_vid_cond and depth_vid is not None:
            fd = self.dv_enc(depth_vid.to(dt)); nt, nh, nw = fd.shape[-3:]
            tc = ((torch.arange(nt, device=dev, dtype=torch.float32) + 0.5) * Fsrc / nt - 0.5)
            ys = (torch.arange(nh, device=dev, dtype=dt) + 0.5) / nh * 2 - 1; xs = (torch.arange(nw, device=dev, dtype=dt) + 0.5) / nw * 2 - 1
            yy, xx = torch.meshgrid(ys, xs, indexing="ij"); yx = torch.stack([yy, xx], -1).reshape(-1, 2)
            ang = yx[..., None] * (2.0 ** torch.arange(self.dv_nf, device=dev, dtype=dt)) * math.pi
            pe_yx = self.dv_xy_pe(torch.cat([yx, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1))       # (nh·nw, H)
            pe = (self._pe_at(tc, dt)[:, None, :] + pe_yx[None]).reshape(nt * nh * nw, -1)
            dtok = self.dv_tok(fd.flatten(2).transpose(1, 2)) + pe[None] + self.geo_type[0]
            dm = torch.ones(B, dtok.shape[1], dtype=torch.bool, device=dev)
            if self.training and self.dv_dropout > 0:
                dm = dm & (torch.rand(B, 1, device=dev) >= self.dv_dropout)
            gparts.append(dtok); gmasks.append(dm)
        if self.geo_latent_dim > 0 and geo_emb is not None:
            Fc = geo_emb.shape[1]
            stok = self.sc_in(geo_emb.to(dt)) + self._pe_at(torch.arange(Fc, device=dev, dtype=torch.float32), dt)[None] + self.geo_type[1]
            sm = geo_mask.bool() if geo_mask is not None else torch.ones(B, Fc, dtype=torch.bool, device=dev)
            if self.training and self.sc_dropout > 0:
                sm = sm & (torch.rand(B, 1, device=dev) >= self.sc_dropout)
            gparts.append(stok); gmasks.append(sm)
        h_geo, m_geo = torch.cat(gparts, 1), torch.cat(gmasks, 1)
        # timestep
        te = self.time_proj(self.time_mlp(t))
        if te.dim() == 3:
            te = te.mean(1)
        h_cam, _, _, _ = self.layers(modality_tokens=(h_cam, h_txt, h_vid, h_geo), modality_masks=(None, m_txt, m_vid, m_geo), time_cond=te)
        return self.out(h_cam)
