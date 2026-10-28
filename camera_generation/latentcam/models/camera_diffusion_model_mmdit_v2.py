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
[R582] 4D 점 구름 조건 (pc_cond, 기본 off = R580 과 state_dict·동작 동일):
  ④ 기하 stream 에 정적 점 토큰 (scene_points4d 의 FPS 앞 N 점 → xyz Fourier MLP + 종류, 시간 PE 없음) 과
     동적 점 토큰 (프레임마다 Nd 점 → 점 MLP → 학습 query Q 개 attention pool → 시간 PE(f) + 종류) 을 넣는다.
     정적·동적은 각각 dropout mask. depth 영상은 depth_vid_cond=false 로 뺀다.
  video_point_pe: Molmo2 패치 토큰 (프레임 f, 8x8) 에 그 패치 3D 점 (patch_points_8x8) Fourier PE 를 pool 전에 더한다 (마지막 층 0 초기화).
  좌표는 dataset 이 frame-s 카메라 / norm_scale 로 바꿔 준다 (track·box 와 같은 게이지).
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
                 depth_vid_cond=True, depth_vid_dropout=0.1, srccam_dropout=0.1, dropout=0.1,
                 pc_cond=False, pc_dyn_queries=8, pc_dropout=0.1, video_point_pe=False,
                 depth_vid_dense=False, depth_vid_dense_depth=True, depth_vid_dense_plucker=True, depth_vid_dense_q=16, **_unused):
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
        self.dv_dense = None
        if self.depth_vid_cond and depth_vid_dense:      # [R588] Molmo2 378² 좌표 dense depth + 픽셀 Plücker → 시간칸 13 × Q 토큰 (Conv3d 대신)
            from models.dense_depth_encoder import DenseDepthEncoder
            self.dv_dense = DenseDepthEncoder(hidden_dim, num_heads, use_depth=bool(depth_vid_dense_depth), use_plucker=bool(depth_vid_dense_plucker),
                                              queries=int(depth_vid_dense_q))
        elif self.depth_vid_cond:
            self.dv_enc = nn.Sequential(nn.Conv3d(2, 32, 3, 2, 1), nn.SiLU(), nn.Conv3d(32, 64, 3, 2, 1), nn.SiLU(),
                                        nn.Conv3d(64, 128, 3, 2, 1), nn.SiLU())
            self.dv_tok = nn.Linear(128, hidden_dim)
            self.dv_xy_pe = nn.Sequential(nn.Linear(2 + 2 * 2 * self.dv_nf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.geo_latent_dim = int(geo_latent_dim or 0)
        if self.geo_latent_dim > 0:
            self.sc_in = nn.Sequential(nn.Linear(self.geo_latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.geo_type = nn.Parameter(torch.zeros(2, hidden_dim))            # [depth, srccam]
        self.geo_null = nn.Parameter(torch.zeros(1, 1, hidden_dim))        # 기하 조건이 전부 빠졌을 때 자리 (mask 전부 False 방지)
        # [R582] 4D 점 구름 (정적·동적) · Molmo2 패치 3D PE
        self.pc_cond, self.pc_dq_n, self.pc_dropout, self.video_point_pe = bool(pc_cond), int(pc_dyn_queries), float(pc_dropout), bool(video_point_pe)
        if self.pc_cond or self.video_point_pe:
            self.register_buffer("pc_freq", 2.0 ** torch.arange(-4, 4, dtype=torch.float32), persistent=False)
            fdim = 3 + 3 * 2 * 8
        if self.pc_cond:
            self.pc_mlp_s = nn.Sequential(nn.Linear(fdim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.pc_mlp_d = nn.Sequential(nn.Linear(fdim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.pc_dq = nn.Parameter(torch.randn(self.pc_dq_n, hidden_dim) * 0.02)
            self.pc_dpool = CrossAttention(hidden_dim, num_heads)
            self.pc_dpool_ln = nn.LayerNorm(hidden_dim)
            self.pc_dempty = nn.Parameter(torch.zeros(1, 1, hidden_dim))    # 동적 점이 없는 프레임도 key 하나는 보이게
            self.pc_type = nn.Parameter(torch.zeros(2, hidden_dim))         # [정적, 동적]
        if self.video_point_pe:
            self.vp_mlp = nn.Sequential(nn.Linear(fdim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            nn.init.zeros_(self.vp_mlp[-1].weight); nn.init.zeros_(self.vp_mlp[-1].bias)
            self.vp_none = nn.Parameter(torch.zeros(hidden_dim))           # depth 없는 패치
        self.layers = MMDiT(depth=num_layers, dim_modalities=(hidden_dim,) * 4, dim_cond=hidden_dim, qk_rmsnorm=True, num_residual_streams=1,
                            heads=num_heads)
        self.out = nn.Linear(hidden_dim, cam_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # train/eval 스크립트가 읽는 계측 속성 (DiT 판과 이름 맞춤)
        self.text_cross_attn_weight = self.geo_cross_attn_weight = self.video_cross_attn_weight = None
        self.readout = None; self.readout_aux_pred = None

    def _fourier(self, p):
        p = p.clamp(-50, 50)
        ang = p[..., None] * self.pc_freq.to(p.dtype) * math.pi
        return torch.cat([p, ang.sin().flatten(-2), ang.cos().flatten(-2)], -1)

    def _pe_at(self, pos, dtype):
        return positional_encoding_at(pos, self.hidden_dim).to(dtype)

    def forward(self, x_t, t, text_emb, text_mask, geo_emb=None, geo_mask=None, cond=None, video_emb=None, video_mask=None,
                video_text_emb=None, video_text_mask=None, depth_vid=None, pc_static=None, pc_static_valid=None, pc_dyn=None,
                pc_dyn_valid=None, video_points=None, video_points_valid=None, dv_cam=None, **_unused):
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
            if self.video_point_pe and video_points is not None:     # [R582] 패치 3D 위치
                assert video_points.shape[1] * video_points.shape[2] == Fv * self.vp, f"video_points {tuple(video_points.shape)} vs Molmo2 {Fv}x{self.vp}"
                pe = self.vp_mlp(self._fourier(video_points.to(dt).reshape(B * Fv, self.vp, 3)))
                vval = video_points_valid.reshape(B * Fv, self.vp, 1).bool()
                v = v + torch.where(vval, pe, self.vp_none.to(dt).expand_as(pe))
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
        if self.dv_dense is not None and (depth_vid is not None or dv_cam is not None):     # [R588]
            dtok, slot = self.dv_dense(depth_vid.to(dt) if depth_vid is not None else None, dv_cam)
            dtok = dtok + self._pe_at(slot.float() * (Fsrc - 1) / max(T - 1, 1), dt)[None] + self.geo_type[0]   # 카메라 토큰과 같은 시간축 (소스 프레임 번호)
            dm = torch.ones(B, dtok.shape[1], dtype=torch.bool, device=dev)
            if self.training and self.dv_dropout > 0:
                dm = dm & (torch.rand(B, 1, device=dev) >= self.dv_dropout)
            gparts.append(dtok); gmasks.append(dm)
        elif self.depth_vid_cond and depth_vid is not None:
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
        if self.pc_cond and pc_static is not None:                     # [R582] 정적 점: 시간 PE 없음
            stok = self.pc_mlp_s(self._fourier(pc_static.to(dt))) + self.pc_type[0]
            spm = pc_static_valid.bool()
            if self.training and self.pc_dropout > 0:
                spm = spm & (torch.rand(B, 1, device=dev) >= self.pc_dropout)
            gparts.append(stok); gmasks.append(spm)
        if self.pc_cond and pc_dyn is not None:                        # [R582] 동적 점: 프레임마다 Q 토큰 + 시간 PE
            Td, Nd = pc_dyn.shape[1], pc_dyn.shape[2]; H = self.hidden_dim; Q = self.pc_dq_n
            dfe = self.pc_mlp_d(self._fourier(pc_dyn.to(dt))).reshape(B * Td, Nd, H)
            kv = torch.cat([self.pc_dempty.to(dt).expand(B * Td, 1, H), dfe], 1)
            kpm = ~torch.cat([torch.ones(B * Td, 1, dtype=torch.bool, device=dev), pc_dyn_valid.reshape(B * Td, Nd).bool()], 1)
            dtok = self.pc_dpool_ln(self.pc_dpool(self.pc_dq.to(dt)[None].expand(B * Td, -1, -1), kv, key_padding_mask=kpm)).reshape(B, Td * Q, H)
            dtok = dtok + self._pe_at(torch.arange(Td, device=dev, dtype=torch.float32).repeat_interleave(Q), dt)[None] + self.pc_type[1]
            dpm = pc_dyn_valid.bool().any(-1).repeat_interleave(Q, 1)
            if self.training and self.pc_dropout > 0:
                dpm = dpm & (torch.rand(B, 1, device=dev) >= self.pc_dropout)
            gparts.append(dtok); gmasks.append(dpm)
        h_geo, m_geo = torch.cat(gparts, 1), torch.cat(gmasks, 1)
        # timestep
        te = self.time_proj(self.time_mlp(t))
        if te.dim() == 3:
            te = te.mean(1)
        h_cam, _, _, _ = self.layers(modality_tokens=(h_cam, h_txt, h_vid, h_geo), modality_masks=(None, m_txt, m_vid, m_geo), time_cond=te)
        return self.out(h_cam)
