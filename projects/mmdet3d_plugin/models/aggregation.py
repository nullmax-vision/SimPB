import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import ATTENTION, PLUGIN_LAYERS


class ReWeight(nn.Module):
    def __init__(self, c_dim, f_dim=256, trans=True, with_pos=False):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.trans = trans
        self.with_pos = with_pos

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.alpha = nn.Sequential(
            nn.Linear(f_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, query, query_pos, parameter, trans_matrix=None):

        alpha = self.alpha(self.reduce(parameter))

        if self.trans:
            reweight_matrix = (trans_matrix * alpha).permute(0, 2, 1)
            reweight_divisor = torch.clamp(reweight_matrix.sum(-1).unsqueeze(-1), 1e-5)
            query = torch.div(torch.matmul(reweight_matrix, query), reweight_divisor)
            query_pos = torch.div(torch.matmul(reweight_matrix, query_pos), reweight_divisor) if self.with_pos else None
        else:
            query = alpha * query
            query_pos = alpha * query_pos if self.with_pos else None

        return query, query_pos


@PLUGIN_LAYERS.register_module()
class AdaptiveQueryAggregation(nn.Module):
    def __init__(self, self_attn=None, reweight=None, decouple_attn=False, with_pos=False):
        super().__init__()
        self.with_pos = with_pos
        self.decouple_attn = decouple_attn
        trans = True if self_attn is not None else False
        self.reweight = ReWeight(c_dim=257, trans=trans, with_pos=with_pos) if reweight is not None else None
        self.self_attn = build_from_cfg(self_attn, ATTENTION) if self_attn is not None else None


    def forward(self,
                query2d, query_pos2d, anchor2d,
                query3d, query_pos3d, anchor3d,
                dn_query2d=None, dn_query_pos2d=None, dn_anchor2d=None,
                dn_query3d=None, dn_query_pos3d=None, dn_anchor3d=None,
                trans_matrix=None, center_matrix=None, dn_trans_matrix=None, dn_center_matrix=None,
                attn_mask=None, graph_model=None, **kwargs):

        if self.reweight is not None:
            center_param = torch.cat([query2d, center_matrix.sum(-1).unsqueeze(-1)], dim=-1)
            query3d_from2d, query_pos3d_from2d = self.reweight(query2d, query_pos2d, center_param, trans_matrix)

            if dn_query2d is not None:
                dn_center_param = torch.cat([dn_query2d, dn_center_matrix.sum(-1).unsqueeze(-1)], dim=-1)
                dn_query3d_from2d, dn_query_pos3d_from2d = self.reweight(dn_query2d, dn_query_pos2d, dn_center_param, dn_trans_matrix)

        else:
            trans_matrix_t = trans_matrix.permute(0, 2, 1)
            trans_divisor = torch.clamp(trans_matrix_t.sum(-1).unsqueeze(-1), 1e-5)
            query3d_from2d = torch.div(torch.matmul(trans_matrix_t, query2d), trans_divisor)
            query_pos3d_from2d = torch.div(torch.matmul(trans_matrix_t, query_pos2d), trans_divisor) if self.with_pos else None

            if dn_query2d is not None:
                query3d_from2d = torch.div(torch.matmul(trans_matrix_t, query2d), trans_divisor)
                query_pos3d_from2d = torch.div(torch.matmul(trans_matrix_t, query_pos2d), trans_divisor) if self.with_pos else None

        # merge with denoise
        if dn_query3d is not None:
            query3d = torch.cat([query3d, dn_query3d], dim=1)
            query_pos3d = torch.cat([query_pos3d, dn_query_pos3d], dim=1)
            anchor3d = torch.cat([anchor3d, dn_anchor3d], dim=1)

            if dn_query2d is not None:
                query3d_from2d = torch.cat([query3d_from2d, dn_query3d_from2d], dim=1)
                query_pos3d_from2d = torch.cat([query_pos3d_from2d, dn_query_pos3d_from2d], dim=1) if self.with_pos else None
            else:
                query3d_from2d = torch.cat([query3d_from2d, torch.zeros_like(dn_query3d)], dim=1)
                query_pos3d_from2d = torch.cat([query_pos3d_from2d, torch.zeros_like(dn_query3d)], dim=1) if self.with_pos else None

        query3d = query3d + query3d_from2d
        query_pos3d = query_pos3d + query_pos3d_from2d if self.with_pos else query_pos3d

        aggregated_query3d = graph_model(self.self_attn,
                                         query=query3d,
                                         query_pos=query_pos3d,
                                         attn_mask=attn_mask)

        return aggregated_query3d, query_pos3d, anchor3d
