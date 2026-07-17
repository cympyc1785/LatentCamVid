import numpy as np
import torch
import torch.nn as nn
import concerto
from concerto.structure import Point
from torch.nn.utils.rnn import pad_sequence
from concerto.utils import batch2offset

import numpy as np

import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, ROOT)
from tools.Mosaic3D.src.infer import load_mosaic3d_model, infer_mosaic3d, get_dummy_input
from concerto.model import GridPooling

class PCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.backbone = load_mosaic3d_model()
        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        self.down1 = GridPooling(
            in_channels=768,
            out_channels=768,
            stride=2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.down2 = GridPooling(
            in_channels=768,
            out_channels=768,
            stride=2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

    def forward(self, point_obj):
        for key in point_obj.keys():
            if isinstance(point_obj[key], torch.Tensor):
                point_obj[key] = point_obj[key].cuda(non_blocking=True)
        point_obj['grid_size'] = 0.02
        point = Point(point_obj)
        out = self.backbone(point)
        point = out['point']
        # point = self.down2(self.down1(point))
        feat = out['clip_feat']
        # print(point['batch'].shape)
        # print(feat.shape)
        # del out
        offset = batch2offset(point['batch'])
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