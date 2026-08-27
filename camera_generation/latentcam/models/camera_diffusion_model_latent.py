import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim):
    """Sinusoidal embedding. t may be (B,) or (B, T) (per-token) -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half) / half
    ).to(t)
    args = t[..., None].float() * freqs                     # (..., half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

def positional_encoding(seq_len, dim, device='cpu'):
    pos = torch.arange(seq_len, device=device).float()[:, None]  # (N, 1)
    i = torch.arange(dim, device=device).float()[None, :]  # (1, D)
    angle_rates = 1 / (10000 ** (2 * (i//2) / dim))
    angle_rads = pos * angle_rates
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return pe  # (N, D)

class FusedMLP(nn.Sequential):
    def __init__(
        self,
        dim_model: int,
        dropout: float,
        activation: nn.Module,
        hidden_layer_multiplier: int = 4,
        bias: bool = True,
    ):
        super().__init__(
            nn.Linear(dim_model, dim_model * hidden_layer_multiplier, bias=bias),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_model * hidden_layer_multiplier, dim_model, bias=bias),
        )

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim)

    def forward(self, t):
        t = timestep_embedding(t, self.dim)
        x = self.lin1(t)
        x = F.silu(x)
        x = self.lin2(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_weight = None

    def forward(self, x, context, attn_mask=None, key_padding_mask=None):
        a, attn_weight = self.attn(x, context, context, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        self.attn_weight = attn_weight
        return self.norm(x + a)
    
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.attn_weight = None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        a, attn_weight = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        if self.training is False:
            self.attn_weight = attn_weight
        return self.norm(x + a)


class CameraDiffusionModel(nn.Module):
    def __init__(
        self,
        cam_dim=11,
        time_emb_dim=512,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        geo_latent_dim=768,
        text_dim=4096,
        dropout=0.1,
        geo_encoder=None,
        geo_cam_raw_dim=0,
        geo_cam_embed_dim=128,
        cond_dim=0,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads

        # [new 2026-08-27] cond_dim > 0 (cfg.target_track_dim): x_t 채널에 per-token 조건
        # (subject OBB track 등)을 concat 한다. concat 조건은 cross-attn 과 달리 uncond
        # forward 에서 "빼기"가 불가능하므로 null = 전 채널 0 (forward 의 cond=None 분기).
        # 0 = 기존 구조 그대로 (기존 ckpt 와 state_dict 호환).
        self.cond_dim = int(cond_dim)
        self.cam_in = nn.Linear(cam_dim + self.cond_dim, hidden_dim)

        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)
        self.mod1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mod2 = nn.Linear(hidden_dim, hidden_dim * 2)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # [new] geo_cam_raw_dim > 0 (cfg.geo_cam_embed): geo_emb arrives as
        # (B, M, geo_latent_dim + geo_cam_raw_dim) — the trailing raw dims are the per-context-view
        # camera pose relative to the target's first camera, broadcast to that view's patches by
        # the training loop. They are split off here and lifted by a TRAINABLE MLP before being
        # concatenated back, so the frozen LagerNVS part stays cacheable. 0 = the original layout.
        self.geo_cam_raw_dim = int(geo_cam_raw_dim)
        if self.geo_cam_raw_dim > 0:
            self.geo_cam_mlp = nn.Sequential(
                nn.Linear(self.geo_cam_raw_dim, geo_cam_embed_dim), nn.SiLU(),
                nn.Linear(geo_cam_embed_dim, geo_cam_embed_dim))
            self.geo_proj = nn.Linear(geo_latent_dim + geo_cam_embed_dim, hidden_dim)
        else:
            self.geo_proj = nn.Linear(geo_latent_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim), 
                SelfAttention(hidden_dim, num_heads),
                CrossAttention(hidden_dim, num_heads),
                FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4),
                nn.LayerNorm(hidden_dim),
                CrossAttention(hidden_dim, num_heads),
                FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4),
            ])
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, cam_dim)
        self.text_cross_attn_weight = None
        self.geo_cross_attn_weight = None

        self.geo_encoder = geo_encoder

    def _lift_geo_cam(self, geo_emb):
        """[new] split the trailing raw camera dims off geo_emb and replace them with the MLP
        embedding. Identity when geo_cam_raw_dim == 0 (the original geo condition)."""
        if self.geo_cam_raw_dim <= 0:
            return geo_emb
        d = self.geo_cam_raw_dim
        return torch.cat([geo_emb[..., :-d], self.geo_cam_mlp(geo_emb[..., -d:])], dim=-1)

    def forward(self, x_t, t, text_emb, text_mask, geo_emb=None, geo_mask=None, cond=None):
        self.text_cross_attn_weight = None
        # geo latent is provided externally (on-the-fly frozen geo_encoder in the
        # training loop); condition on it whenever geo_emb is given.
        has_geo_latent = geo_emb is not None
        if has_geo_latent:
            self.geo_cross_attn_weight = None
        B, T, _ = x_t.shape

        device = x_t.device

        # [new 2026-08-27] per-token concat 조건. cond=None 이면 zeros = null 조건 — 계측용
        # 호출(geo_attn_probe 등)이 cond 를 모르고 불러도 죽지 않게 하려는 것이지, 학습/샘플링
        # 경로는 반드시 cond 를 명시적으로 넘겨야 한다 (train_latent_cam_dm.build_track_cond).
        if self.cond_dim > 0:
            if cond is None:
                cond = x_t.new_zeros(B, T, self.cond_dim)
            x_t = torch.cat([x_t, cond.to(x_t.dtype)], dim=-1)

        h = self.cam_in(x_t)

        pos_emb = positional_encoding(T, h.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
        h = h + pos_emb

        # t: (B,) scalar-per-sample (legacy, broadcast over tokens) OR (B,T) per-token noise.
        t_embed = self.time_mlp(t)              # (B,D) or (B,T,D)
        t_embed = self.time_proj(t_embed)       # (B,hidden) or (B,T,hidden)
        per_token = (t_embed.dim() == 3)        # per-token FiLM vs broadcast

        text_tok = self.text_proj(text_emb)
        text_tok = text_tok + positional_encoding(text_tok.shape[-2], text_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        if has_geo_latent:
            geo_tok = self.geo_proj(self._lift_geo_cam(geo_emb))
            geo_tok = geo_tok * geo_mask.unsqueeze(-1)

        for norm1, self_attn, text_cross_attn, mlp1, norm2, geo_cross_attn, mlp2 in self.layers:

            scale1, shift1 = self.mod1(t_embed).chunk(2, dim=-1)
            if not per_token:                          # (B,D)->(B,1,D) broadcast (legacy)
                scale1 = scale1.unsqueeze(1)
                shift1 = shift1.unsqueeze(1)           # per-token: (B,T,D) element-wise

            h = norm1(h)
            h = h * (1 + scale1) + shift1

            h = self_attn(h)
            h = text_cross_attn(h, text_tok, key_padding_mask=~text_mask)
            h = mlp1(h) + h

            if has_geo_latent:
                scale2, shift2 = self.mod2(t_embed).chunk(2, dim=-1)
                if not per_token:
                    scale2 = scale2.unsqueeze(1)  # (B,1,D)
                    shift2 = shift2.unsqueeze(1)

                h = norm2(h)
                h = h * (1 + scale2) + shift2

                h = geo_cross_attn(h, geo_tok, key_padding_mask=~geo_mask)
                h = mlp2(h) + h

            if self.training is False:
                if self.text_cross_attn_weight is None:
                    self.text_cross_attn_weight = text_cross_attn.attn_weight
                    if has_geo_latent:
                        self.geo_cross_attn_weight = geo_cross_attn.attn_weight
                else:
                    self.text_cross_attn_weight += text_cross_attn.attn_weight
                    if has_geo_latent:
                        self.geo_cross_attn_weight += geo_cross_attn.attn_weight
            
            
        if self.training is False:
            self.text_cross_attn_weight /= len(self.layers)
            if has_geo_latent:
                self.geo_cross_attn_weight /= len(self.layers)

        h = h[:, :T, :]
        return self.out(h)# (B, T, 9)

    def forward_ar(self, x_t, t, text_emb, text_mask, geo_emb=None, geo_mask=None, history=None):
        """Chunk-wise AR: denoise the current chunk x_t (B, Cc, cam_dim) conditioned on
        past CLEAN latents `history` (B, Ph, cam_dim) via CAUSAL self-attention over the
        concatenated sequence [history | current], plus text/geo cross-attn. Diffusion
        time modulation is applied to the current-chunk positions only (history is clean).
        Returns the predicted noise for the CURRENT chunk only (B, Cc, cam_dim)."""
        B, Cc, _ = x_t.shape
        device = x_t.device
        if history is not None and history.shape[1] > 0:
            seq = torch.cat([history, x_t], dim=1)          # (B, Ph+Cc, cam_dim)
            n_hist = history.shape[1]
        else:
            seq = x_t
            n_hist = 0
        L = seq.shape[1]

        h = self.cam_in(seq)
        h = h + positional_encoding(L, h.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        t_embed = self.time_proj(self.time_mlp(t))
        text_tok = self.text_proj(text_emb)
        text_tok = text_tok + positional_encoding(text_tok.shape[-2], text_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        has_geo_latent = geo_emb is not None
        if has_geo_latent:
            geo_tok = self.geo_proj(self._lift_geo_cam(geo_emb)) * geo_mask.unsqueeze(-1)

        # causal self-attn mask: position i attends to positions <= i
        causal = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)
        # time modulation applies to current chunk positions only (history is clean)
        mod_pos = torch.zeros(1, L, 1, device=device)
        mod_pos[:, n_hist:, :] = 1.0

        for norm1, self_attn, text_cross_attn, mlp1, norm2, geo_cross_attn, mlp2 in self.layers:
            s1, sh1 = self.mod1(t_embed).chunk(2, dim=-1)
            s1 = s1.unsqueeze(1); sh1 = sh1.unsqueeze(1)
            h = norm1(h)
            h = h * (1 + s1 * mod_pos) + sh1 * mod_pos
            h = self_attn(h, attn_mask=causal)
            h = text_cross_attn(h, text_tok, key_padding_mask=~text_mask)
            h = mlp1(h) + h
            if has_geo_latent:
                s2, sh2 = self.mod2(t_embed).chunk(2, dim=-1)
                s2 = s2.unsqueeze(1); sh2 = sh2.unsqueeze(1)
                h = norm2(h)
                h = h * (1 + s2 * mod_pos) + sh2 * mod_pos
                h = geo_cross_attn(h, geo_tok, key_padding_mask=~geo_mask)
                h = mlp2(h) + h

        return self.out(h[:, n_hist:])                      # current chunk only

def get_model():
    model = CameraDiffusionModel()
    return model