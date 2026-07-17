import numpy as np
import torch
import torch.nn as nn
import concerto
from concerto.structure import Point
from torch.nn.utils.rnn import pad_sequence
from concerto.utils import batch2offset
from torch_scatter import scatter_mean, scatter_max
import copy

import numpy as np

class PCEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        config = [
            dict(type="CenterShift", apply_z=True),
            dict(type="Update", keys_dict={"index_valid_keys": ["coord", "color", "normal", "batch"]}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "color", "inverse", "batch"),
                feat_keys=("coord", "color", "normal"),
            ),
        ]
        self.pc_transform = concerto.transform.Compose(config)
        self.model = concerto.model.load("concerto_large_outdoor", repo_id="Pointcept/Concerto").cuda()
        self.point = None

    def get_p2v_map(self, point:Point):
        inverse_list = []
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            inverse_list.append(inverse)
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            inverse_list.append(inverse)
            parent.feat = point.feat[inverse]
            point = parent
        inverse_list.reverse()
        p2v_map = inverse_list[0]
        for inv in inverse_list[1:]:
            p2v_map = inv[p2v_map]
        
        batch = point['batch']
        # is_monotonic = torch.all(batch[1:] >= batch[:-1])
        offset = batch2offset(batch)
        start = torch.cat([offset.new_zeros(1), offset[:-1]])
        p2v_map_list = [
            p2v_map[start[i]:offset[i]]
            for i in range(len(offset))
        ]
        p2v_map_list = [p2v_map_list[i] - (p2v_map_list[i-1].max() + 1) if i != 0 else p2v_map_list[i] for i in range(len(p2v_map_list))]

        return p2v_map_list

    def decode(self, point:Point, recover_origin_scale=False, num_iter=2):
        for _ in range(num_iter):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        if num_iter == 2 or recover_origin_scale:
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            if recover_origin_scale:
                feat = point.feat[point.inverse]
            else:
                feat = point.feat
        else:
            feat = point.feat
        return feat, point

    def decode_with_attention(self, point:Point, attention, recover_origin_scale=False, num_iter=2):
        """
        attention: (N, 1)
        """
        ret_attention = attention
        for _ in range(num_iter):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            ret_attention = ret_attention[inverse]
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            ret_attention = ret_attention[inverse]
            point = parent
        if recover_origin_scale:
            feat = point.feat[point.inverse]
            ret_attention = ret_attention[point.inverse]
        else:
            feat = point.feat
        return feat, point, ret_attention

    def forward(self, point_obj, ray_info_list=None):
        """
        ray_info_list = List of (ray_sim, visibility in first camera)
        """
        for key in point_obj.keys():
            if isinstance(point_obj[key], torch.Tensor):
                point_obj[key] = point_obj[key].cuda(non_blocking=True)
        point = Point(point_obj)
        point = self.model(point)

        p2v_map_list = self.get_p2v_map(copy.deepcopy(point))

        visibility_in_first_cam = [scatter_max(ray_info[1], p2v_map, dim=0)[0].unsqueeze(-1) for ray_info, p2v_map in zip(ray_info_list, p2v_map_list)]
        ray_sim_list = [scatter_mean(ray_info[0], p2v_map, dim=1).transpose(0, 1) for ray_info, p2v_map in zip(ray_info_list, p2v_map_list)]

        # Decode
        # _, point = self.decode(point, num_iter=1)

        feat = point.feat
        offset = batch2offset(point['batch'])
        start = torch.cat([offset.new_zeros(1), offset[:-1]])

        pc_embed_list = [
            feat[start[i]:offset[i]]
            for i in range(len(offset))
        ]

        pc_embed_list = [torch.cat([vis, pc_embed], dim=1) for vis, pc_embed in zip(visibility_in_first_cam, pc_embed_list)]

        pc_embeds = pad_sequence(pc_embed_list, batch_first=True)  # (B, N_max, D)
        ray_sims = pad_sequence(ray_sim_list, batch_first=True)

        lengths = torch.tensor([x.shape[0] for x in pc_embed_list])
        masks = torch.arange(pc_embeds.shape[1])[None, :] < lengths[:, None]
        masks = masks.to(pc_embeds).bool()

        return pc_embeds, ray_sims, p2v_map_list, masks, point # (B, M, 768), (B, M, 1), (B, M), Point
        return pc_embeds, masks, point # (B, M, 768), (B, M, 1), (B, M), Point