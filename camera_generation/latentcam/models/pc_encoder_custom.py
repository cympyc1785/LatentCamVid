import numpy as np
import torch
import torch.nn as nn
import concerto
from concerto.structure import Point
from concerto.model import GridPooling, Embedding, LayerScale
from concerto.module import PointModule, PointSequential
from torch.nn.utils.rnn import pad_sequence
from concerto.utils import batch2offset
from timm.layers import DropPath
import time

import numpy as np

class MLPBlock(PointModule):
    def __init__(
        self,
        channels,
        mlp_ratio=4.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.norm = PointSequential(norm_layer(channels))

        hidden_dim = int(channels * mlp_ratio)

        self.mlp = PointSequential(
            nn.Linear(channels, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, channels),
        )

        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0 else nn.Identity()
        )

        if layer_scale is not None:
            self.ls = PointSequential(LayerScale(channels, layer_scale))
        else:
            self.ls = PointSequential(nn.Identity())

    def forward(self, point: Point):
        shortcut = point.feat

        if self.pre_norm:
            point = self.norm(point)

        point = self.mlp(point)
        point = self.ls(point)
        point = self.drop_path(point)

        point.feat = shortcut + point.feat

        if not self.pre_norm:
            point = self.norm(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)

        return point
    
class PCEncoder(PointModule):
    def __init__(
        self,
        in_channels=6,
        # stride=(2, 2, 2, 2, 2),
        # enc_depths=(2, 2, 2, 2, 2, 2),
        # enc_channels=(32, 64, 128, 256, 512, 1024),
        # stride=(2, 2, 2, 2),
        # enc_depths=(3, 3, 3, 3, 3),
        # enc_channels=(64, 128, 256, 384, 512),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 2, 2),
        enc_channels=(64, 128, 256, 384, 512),
        mlp_ratio=4.0,
        drop_path=0.1,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
    ):
        super().__init__()

        self.num_stages = len(enc_depths)
        self.shuffle_orders = shuffle_orders

        # normalization / activation
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        # embedding
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        # stochastic depth schedule
        enc_drop_path = torch.linspace(0, drop_path, sum(enc_depths)).tolist()

        self.enc = PointSequential()

        dp_offset = 0

        for s in range(self.num_stages):
            stage = PointSequential()

            # downsample
            if s > 0:
                stage.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )

            # blocks
            for i in range(enc_depths[s]):
                stage.add(
                    MLPBlock(
                        channels=enc_channels[s],
                        mlp_ratio=mlp_ratio,
                        drop_path=enc_drop_path[dp_offset + i],
                        layer_scale=layer_scale,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                    ),
                    name=f"block{i}",
                )

            dp_offset += enc_depths[s]

            self.enc.add(stage, name=f"stage{s}")
        
        # self.hidden_dim = enc_channels[-1]
        # self.learnable_embed = nn.Parameter(torch.randn(13, self.hidden_dim))
        # self.cam_in = nn.Linear(32, self.hidden_dim)
        # self.pos_encod = PositionalEncoding(self.hidden_dim, dropout=0.1, batch_first=True)
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, point_obj):

        for key in point_obj.keys():
            if isinstance(point_obj[key], torch.Tensor):
                point_obj[key] = point_obj[key].cuda(non_blocking=True)
        point = Point(point_obj)

        point = self.embedding(point)
        point.serialization(order="z", shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)

        feat = point.feat
        offset = point['offset']
        start = torch.cat([offset.new_zeros(1), offset[:-1]])
        pc_embed_list = [
            feat[start[i]:offset[i]]
            for i in range(len(offset))
        ]
        pc_embeds = pad_sequence(pc_embed_list, batch_first=True)  # (B, N_max, D)

        lengths = torch.tensor([x.shape[0] for x in pc_embed_list])
        masks = torch.arange(pc_embeds.shape[1])[None, :] < lengths[:, None]
        masks = masks.to(pc_embeds).bool()
        return pc_embeds, masks, point # (B, M, 768), (B, M), Point

    # def decode_with_embedding(self, cam_latents, pc_embeds, pc_masks):
    #     """
    #     cam_latents: (B, T, D)
    #     pc_embeds: (B, M, D')
    #     pc_masks: (B, M)
    #     """
    #     B, _, _ = cam_latents.shape[0]

    #     cam = self.cam_in(cam_latents)
    #     cam = self.pos_encod(cam)
    #     cam_masks = torch.ones(cam.shape[:2]).bool()

    #     image_embeds = self.learnable_embed.expand(B, -1)
    #     image_masks = torch.ones(image_embeds.shape[:2]).bool()
        
    #     toks = torch.cat([image_embeds, cam, pc_embeds])
    #     tok_masks = torch.cat([image_masks, cam_masks, pc_masks])
    #     out = self.encoder(toks, src_key_padding_mask=~tok_masks)
    #     return out
