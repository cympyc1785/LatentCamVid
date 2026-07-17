import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from concerto.module import PointSequential
import spconv.pytorch as spconv

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half) / half
    ).to(t)
    args = t[:, None] * freqs[None]
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
        if self.training is False:
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
        cam_dim=32,
        time_emb_dim=512,
        hidden_dim=512,
        num_layers=8,
        cross_layers=4,
        num_heads=8,
        point_dim=769,
        text_dim=4096,
        dropout=0.1,
        pc_encoder=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads

        self.cam_in = nn.Linear(cam_dim, hidden_dim)

        self.time_mlp = TimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        self.mm_cross = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim), 
                CrossAttention(hidden_dim, num_heads),
                FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4)
            ])
            for _ in range(cross_layers)
        ])

        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(hidden_dim), 
                SelfAttention(hidden_dim, num_heads),
                CrossAttention(hidden_dim, num_heads),
                FusedMLP(hidden_dim, dropout, nn.GELU, hidden_layer_multiplier=4)
            ])
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(hidden_dim, cam_dim)
        self.text_cross_attn_weight = None
        self.point_cross_attn_weight = None
        self.mm_cross_attn_weight = None

        self.pc_encoder = pc_encoder

    def forward(self, x_t, t, text_emb, text_mask, point_emb=None, point_mask=None):
        has_point = point_emb is not None and point_mask is not None
        B, T, _ = x_t.shape

        device = x_t.device

        h = self.cam_in(x_t)

        pos_emb = positional_encoding(T, h.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
        h = h + pos_emb

        t_embed = self.time_mlp(t)
        t_embed = self.time_proj(t_embed)
        t_embed = t_embed.unsqueeze(1)
        t_mask = torch.ones(t_embed.shape[:2], device=device).bool()
        
        text_tok = self.text_proj(text_emb)
        context = torch.cat([t_embed, text_tok], dim=1)
        context_mask = torch.cat([t_mask, text_mask], dim=1)
        context_pos_emb = positional_encoding(context.shape[-2], context.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
        context = context + context_pos_emb

        if has_point:
            point_tok = self.point_proj(point_emb)
            point_tok = point_tok * point_mask.unsqueeze(-1)
            for norm, ca, ffn in self.mm_cross:
                point_tok = norm(point_tok)
                point_tok = ca(point_tok, context, key_padding_mask=~context_mask)
                point_tok = ffn(point_tok) + point_tok
            # context = torch.cat([context, point_tok], dim=1)
            # context_mask = torch.cat([context_mask, point_mask], dim=1)
            context = torch.cat([context[:, 0:1, :], point_tok], dim=1)
            context_mask = torch.cat([context_mask[:, 0:1], point_mask], dim=1)
            
        
        for norm, self_attn, context_cross_attn, mlp in self.layers:
            h = norm(h)
            h = self_attn(h)
            h = context_cross_attn(h, context, key_padding_mask=~context_mask)
            h = mlp(h) + h

        h = h[:, :T, :]
        return self.out(h)   # (B, T, 9)
    
def get_model():
    model = CameraDiffusionModel()
    return model