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
        num_heads=8,
        point_dim=772,
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
        self.mod1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.mod2 = nn.Linear(hidden_dim, hidden_dim * 2)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.point_proj = nn.Linear(point_dim, hidden_dim)

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
        self.point_cross_attn_weight = None

        self.pc_encoder = pc_encoder

    def forward(self, x_t, t, text_emb, text_mask, point_emb=None, point_mask=None):

        self.text_cross_attn_weight = None
        self.point_cross_attn_weight = None

        B, T, _ = x_t.shape

        device = x_t.device

        h = self.cam_in(x_t)

        pos_emb = positional_encoding(T, h.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)
        h = h + pos_emb

        t_embed = self.time_mlp(t)
        t_embed = self.time_proj(t_embed)
        
        text_tok = self.text_proj(text_emb)
        text_tok = text_tok + positional_encoding(text_tok.shape[-2], text_tok.shape[-1], device=device).unsqueeze(0).expand(B, -1, -1)

        point_tok = self.point_proj(point_emb)
        point_tok = point_tok * point_mask.unsqueeze(-1)
        
        cross_attn_list = []
        for norm1, self_attn, text_cross_attn, mlp1, norm2, point_cross_attn, mlp2 in self.layers:

            scale1, shift1 = self.mod1(t_embed).chunk(2, dim=-1)
            scale1 = scale1.unsqueeze(1)  # (B,1,D)
            shift1 = shift1.unsqueeze(1)

            h = norm1(h)
            h = h * (1 + scale1) + shift1

            h = self_attn(h)
            h = text_cross_attn(h, text_tok, key_padding_mask=~text_mask)
            h = mlp1(h) + h

            scale2, shift2 = self.mod2(t_embed).chunk(2, dim=-1)
            scale2 = scale2.unsqueeze(1)  # (B,1,D)
            shift2 = shift2.unsqueeze(1)
            
            h = norm2(h)
            h = h * (1 + scale2) + shift2

            h = point_cross_attn(h, point_tok, key_padding_mask=~point_mask)
            h = mlp2(h) + h

            cross_attn_list.append(point_cross_attn.attn_weight)
            if self.training is False:
                if self.text_cross_attn_weight is None:
                    self.text_cross_attn_weight = text_cross_attn.attn_weight
                    self.point_cross_attn_weight = point_cross_attn.attn_weight
                else:
                    self.text_cross_attn_weight += text_cross_attn.attn_weight
                    self.point_cross_attn_weight += point_cross_attn.attn_weight
            
            
        if self.training is False:
            self.text_cross_attn_weight /= len(self.layers)
            self.point_cross_attn_weight /= len(self.layers)

        h = h[:, :T, :]
        cross_attn = torch.stack(cross_attn_list)
        return self.out(h), cross_attn   # (B, T, 9)
    
def get_model():
    model = CameraDiffusionModel()
    return model