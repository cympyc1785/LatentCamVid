import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from mmdit.mmdit_generalized_pytorch import MMDiT
import numpy as np

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half) / half
    ).to(t)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000, batch_first=False) -> None:
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)

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

class CameraDiffusionModel(nn.Module):
    def __init__(
        self,
        cam_dim=32,
        time_emb_dim=512,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        point_dim=768,
        text_dim=768,
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
        self.pos_encod = PositionalEncoding(hidden_dim, dropout, batch_first=True)
        self.cond_pos_encod = PositionalEncoding(hidden_dim, dropout, batch_first=True)

        self.layers = MMDiT(
            depth=self.num_layers,
            dim_modalities=(hidden_dim, hidden_dim, hidden_dim),
            dim_cond=hidden_dim,
            qk_rmsnorm=True,
        )

        self.out = nn.Linear(hidden_dim, cam_dim)

        self.pc_encoder = pc_encoder

    def forward(self, x_t, t, text_emb, text_mask, point_emb=None, point_mask=None):
        has_point = (point_emb is not None and point_mask is not None)
        B, T, _ = x_t.shape

        h = self.cam_in(x_t)

        h = self.pos_encod(h)

        t_embed = self.time_mlp(t)
        t_embed = self.time_proj(t_embed)
        
        text_tok = self.text_proj(text_emb)
        text_tok = self.cond_pos_encod(text_tok)

        if has_point:
            point_tok = self.point_proj(point_emb)
        
        h, text_tok, point_tok = self.layers(
            modality_tokens=(h, text_tok, point_tok),
            modality_masks=(None, text_mask, point_mask),
            time_cond=t_embed
        )

        return self.out(h)   # (B, T, 9)
    
def get_model():
    model = CameraDiffusionModel()
    return model