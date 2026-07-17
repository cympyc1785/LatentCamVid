import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from torchtyping import TensorType
import numpy as np
from einops import rearrange
from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)
    
class PositionalEmbedding(nn.Module):
    """
    Taken from https://github.com/NVlabs/edm
    """

    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32)
        freqs = 2 * freqs / self.num_channels
        freqs = (1 / self.max_positions) ** freqs
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x = torch.outer(x, self.freqs)
        out = torch.cat([x.cos(), x.sin()], dim=1)
        return out.to(x.dtype)

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


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class TimeEmbedder(nn.Module):
    def __init__(
        self,
        dim: int,
        time_scaling: float,
        expansion: int = 4,
    ):
        super().__init__()
        self.encode_time = PositionalEmbedding(num_channels=dim, endpoint=True)

        self.time_scaling = time_scaling
        self.map_time = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim * expansion),
        )

    def forward(self, t: Tensor) -> Tensor:
        time = self.encode_time(t * self.time_scaling)
        time_mean = time.mean(dim=-1, keepdim=True)
        time_std = time.std(dim=-1, keepdim=True)
        time = (time - time_mean) / time_std
        return self.map_time(time)


def modulate_shift_and_scale(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale).unsqueeze(1) + shift.unsqueeze(1)


class LayerNorm16Bits(torch.nn.LayerNorm):
    """
    16-bit friendly version of torch.nn.LayerNorm
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-06,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )
        with torch.autocast(enabled=False, device_type=module_device.type):
            return nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )

class StochatichDepth(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.survival_prob = 1.0 - p

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.survival_prob < 1:
            mask = (
                torch.empty(x.shape[0], 1, 1, device=x.device).uniform_()
                + self.survival_prob
            )
            mask = mask.floor()
            if self.survival_prob > 0:
                mask = mask / self.survival_prob
            return x * mask
        else:
            return x
        
class CrossAttentionSABlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        dim_cond: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.0,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.ca = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_cond,
            is_sa=False,
            use_biases=use_biases,
        )
        self.ca_stochastic_depth = StochatichDepth(stochastic_depth)
        self.ca_ln = layer_norm(dim_qkv, eps=1e-6)

        self.initial_ln = layer_norm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim

        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        cond: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ):
        if cond_mask is None:
            cond_attention_mask = None
        else:
            cond_attention_mask = torch.ones(
                cond.shape[0],
                1,
                cond.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ) * cond_mask.unsqueeze(2)
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        ca_output = self.ca(
            self.ca_ln(tokens),
            cond,
            attention_mask=token_mask,
            key_padding_mask=~cond_mask,
        )
        ca_output = torch.nan_to_num(
            ca_output, nan=0.0, posinf=0.0, neginf=0.0
        )  # Needed as some tokens get attention from no token so Nan
        tokens = tokens + self.ca_stochastic_depth(ca_output)
        attention_output = self.sa(
            self.initial_ln(tokens),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                self.layer_scale_2 * self.ffn(self.middle_ln(tokens))
            )
        else:
            tokens = tokens + self.sa_stochastic_depth(attention_output)
            tokens = tokens + self.ffn_stochastic_depth(
                self.ffn(self.middle_ln(tokens))
            )
        return tokens
    
class CrossAttentionOp(nn.Module):
    def __init__(
        self, attention_dim, num_heads, dim_q, dim_kv, use_biases=True, is_sa=False
    ):
        super().__init__()
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.use_biases = use_biases
        self.is_sa = is_sa
        if self.is_sa:
            self.qkv = nn.Linear(dim_q, attention_dim * 3, bias=use_biases)
        else:
            self.q = nn.Linear(dim_q, attention_dim, bias=use_biases)
            self.kv = nn.Linear(dim_kv, attention_dim * 2, bias=use_biases)
        self.out = nn.Linear(attention_dim, dim_q, bias=use_biases)
        self.attn_weight = None

    def forward(self, x_to, x_from=None, attention_mask=None, key_padding_mask=None):
        if x_from is None:
            x_from = x_to
        if self.is_sa:
            q, k, v = self.qkv(x_to).chunk(3, dim=-1)
        else:
            q = self.q(x_to)
            k, v = self.kv(x_from).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
        if key_padding_mask is not None:
            attention_mask = (~key_padding_mask.bool())[:, None, None, :]
            attention_mask = attention_mask.expand(-1, q.size(1), q.size(2), -1)

        if self.training is False:
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            if attention_mask is not None:
                attn_logits = attn_logits.masked_fill(attention_mask == False, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1).mean(dim=1)
            self.attn_weight = attn_weights
        
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x
    
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_qkv: int,
        num_heads: int,
        attention_dim: int = 0,
        mlp_multiplier: int = 4,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        use_biases: bool = True,
        use_layer_scale: bool = False,
        layer_scale_value: float = 0.0,
        use_layernorm16: bool = True,
    ):
        super().__init__()
        layer_norm = LayerNorm16Bits if use_layernorm16 else nn.LayerNorm
        self.initial_ln = layer_norm(dim_qkv, eps=1e-6)
        attention_dim = dim_qkv if attention_dim == 0 else attention_dim
        self.sa = CrossAttentionOp(
            attention_dim,
            num_heads,
            dim_qkv,
            dim_qkv,
            is_sa=True,
            use_biases=use_biases,
        )
        self.sa_stochastic_depth = StochatichDepth(stochastic_depth)
        self.middle_ln = layer_norm(dim_qkv, eps=1e-6)
        self.ffn = FusedMLP(
            dim_model=dim_qkv,
            dropout=dropout,
            activation=nn.GELU,
            hidden_layer_multiplier=mlp_multiplier,
            bias=use_biases,
        )
        self.ffn_stochastic_depth = StochatichDepth(stochastic_depth)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                torch.ones(dim_qkv) * layer_scale_value, requires_grad=True
            )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ):
        if token_mask is None:
            attention_mask = None
        else:
            attention_mask = token_mask.unsqueeze(1) * torch.ones(
                tokens.shape[0],
                tokens.shape[1],
                1,
                dtype=torch.bool,
                device=tokens.device,
            )
        attention_output = self.sa(
            self.initial_ln(tokens),
            attention_mask=attention_mask,
        )
        if self.use_layer_scale:
            tokens = tokens + self.sa_stochastic_depth(
                self.layer_scale_1 * attention_output
            )
            tokens = tokens + self.ffn_stochastic_depth(
                self.layer_scale_2 * self.ffn(self.middle_ln(tokens))
            )
        else:
            tokens = tokens + self.sa_stochastic_depth(attention_output)
            tokens = tokens + self.ffn_stochastic_depth(
                self.ffn(self.middle_ln(tokens))
            )
        return tokens

class CameraDiffusionModel(nn.Module):
    def __init__(
        self,
        cam_dim=32,
        time_emb_dim=512,
        hidden_dim=512,
        num_layers=4,
        mlp_multiplier=4,
        num_heads=8,
        point_dim=1280,
        text_dim=4096,
        dropout=0.1,
        stochastic_depth=0.1,
        device="cuda"
    ):
        super().__init__()

        self.cam_in = nn.Sequential(
            nn.Linear(cam_dim, hidden_dim),
            PositionalEncoding(hidden_dim),
        )
        self.cond_positional_embedding = PositionalEncoding(hidden_dim, max_len=10000)

        self.time_embedding = TimeEmbedder(hidden_dim // 4, time_scaling=1000)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.point_proj = nn.Linear(point_dim, hidden_dim)

        self.use_layernorm16 = device == "cuda"

        self.cond_sa = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim_qkv=hidden_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(2)
            ]
        )
        self.backbone_module = nn.ModuleList(
            [
                CrossAttentionSABlock(
                    dim_qkv=hidden_dim,
                    dim_cond=hidden_dim,
                    num_heads=num_heads,
                    mlp_multiplier=mlp_multiplier,
                    dropout=dropout,
                    stochastic_depth=stochastic_depth,
                    use_layernorm16=self.use_layernorm16,
                )
                for _ in range(num_layers)
            ]
        )

        layer_norm = LayerNorm16Bits if self.use_layernorm16 else nn.LayerNorm
        self.out_norm = layer_norm(hidden_dim, eps=1e-6)
        self.out = nn.Linear(hidden_dim, cam_dim)

        self.text_cross_attn_weight = None
        self.point_cross_attn_weight = None
        self.mm_cross_attn_weight = None

    def forward(self, x_t, t, text_emb, text_mask, point_emb=None, point_mask=None):
        B = x_t.shape[0]
        h = self.cam_in(x_t)
        t = self.time_embedding(t)
        t_mask = torch.ones((B, 1)).bool()

        text_tok = self.text_proj(text_emb)
        if point_emb is not None and point_mask is not None:
            point_tok = self.point_proj(point_emb)
            context = torch.cat([t.unsqueeze(1), text_tok, point_tok], dim=1)
            context_mask = torch.cat([t_mask.to(text_mask), text_mask, point_mask], dim=1)
        else:
            context = torch.cat([t.unsqueeze(1), text_tok], dim=1)
            context_mask = torch.cat([t_mask.to(text_mask), text_mask], dim=1)
        context = self.cond_positional_embedding(context)
        
        for block in self.cond_sa:
            context = block(context, token_mask=context_mask)

            # if self.training is False:
            #     context_attn_weight = block.sa.attn_weight
            #     if self.mm_cross_attn_weight is None:
            #         self.mm_cross_attn_weight = block.sa.attn_weight[:, 1+text_tok.shape[1]:, 1:text_tok.shape[1] + 1]
            #     else:
            #         self.mm_cross_attn_weight += block.sa.attn_weight[:, 1+text_tok.shape[1]:, 1:text_tok.shape[1] + 1]

        for block in self.backbone_module:
            h = block(h, context, cond_mask=context_mask)
            # if self.training is False:
            #     context_attn_weight = block.ca.attn_weight
            #     if self.text_cross_attn_weight is None:
            #         self.text_cross_attn_weight = context_attn_weight[:, :, 1:text_tok.shape[1] + 1]
            #         self.point_cross_attn_weight = context_attn_weight[:, :, text_tok.shape[1] + 1:]
            #     else:
            #         self.text_cross_attn_weight += context_attn_weight[:, :, 1:text_tok.shape[1] + 1]
            #         self.point_cross_attn_weight += context_attn_weight[:, :, text_tok.shape[1] + 1:]
        h = self.out(self.out_norm(h))

        # if self.training is False:
        #     self.text_cross_attn_weight /= len(self.backbone_module)
        #     self.point_cross_attn_weight /= len(self.backbone_module)
        #     self.mm_cross_attn_weight /= len(self.cond_sa)

        return h # (B, T, cam_dim)
    
def get_model():
    model = CameraDiffusionModel()
    return model