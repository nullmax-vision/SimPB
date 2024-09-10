import torch
import torch.nn as nn
import numpy as np

from mmcv.cnn import Linear, Scale, bias_init_with_prob
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
)

from ..blocks import linear_relu_ln
from ..utils import inverse_sigmoid, pos2posemb2d

__all__ = [
    "SparseBox2DRefinementModule",
    "SparseBox2DEncoder",
]

@POSITIONAL_ENCODING.register_module()
class SparseBox2DEncoder(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            with_size=False,
            with_sin_embed=False,
            mode="add",
            in_loops=1,
            out_loops=2,
    ):
        super(SparseBox2DEncoder, self).__init__()
        self.embed_dims = embed_dims
        self.mode = mode
        self.with_size = with_size
        self.with_sin_embed = with_sin_embed

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, in_loops, out_loops, input_dims))

        if self.with_sin_embed:
            self.query_embeddings2d = embedding_layer(256)
        else:
            self.pos_fc = embedding_layer(2)
            if self.with_size:
                self.size_fc = embedding_layer(2)
                self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_2d):
        if self.with_sin_embed:
            output = self.query_embeddings2d(pos2posemb2d(box_2d))
        else:
            pos_feat = self.pos_fc(box_2d[..., :2])
            if self.with_size:
                size_feat = self.size_fc(box_2d[..., 2:4])
                if self.mode == "add":
                    output = pos_feat + size_feat
                elif self.mode == "cat":
                    output = torch.cat([pos_feat, size_feat], dim=-1)
                output = self.output_fc(output)
            else:
                output = pos_feat

        return output

@PLUGIN_LAYERS.register_module()
class SparseBox2DRefinementModule(BaseModule):
    def __init__(self, embed_dims=256, output_dim=4, num_cls=10, alpha_dim=2,
                 with_cls_branch=True, with_alpha_branch=False, with_depth_branch=False,
                 with_multibin_depth=False, depth_bin_num=64):
        super(SparseBox2DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim)
        )

        self.with_cls_branch = with_cls_branch
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, self.num_cls),
            )
        self.with_alpha_branch = with_alpha_branch
        if with_alpha_branch:
            self.alpha_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                Linear(self.embed_dims, alpha_dim),
                Scale([1.0] * 2)
            )
        self.with_depth_branch = with_depth_branch
        self.with_multibin_depth = with_multibin_depth
        if with_depth_branch:
            if with_multibin_depth:
                self.depth_layers = nn.Sequential(
                    *linear_relu_ln(embed_dims, 2, 2),
                    Linear(self.embed_dims, depth_bin_num),
                )
            else:
                self.depth_layers = nn.Sequential(
                    *linear_relu_ln(embed_dims, 2, 2),
                    Linear(self.embed_dims, 1),
                    Scale([1.0] * 1)
                )

    def init_weight(self):
        if self.with_cls_branch:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)
        if self.with_multibin_depth:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.depth_layers[-1].bias, bias_init)

    def forward(self, instance_feature, anchor2d, anchor2d_embed,
                metas=None, return_cls=True, query_groups=None):
        output = self.layers(instance_feature + anchor2d_embed)

        if anchor2d.shape[-1] == 2:
            output[..., :2] = output[..., :2] + inverse_sigmoid(anchor2d)
        elif anchor2d.shape[-1] == 4:
            output[..., :4] = output[..., :4] + inverse_sigmoid(anchor2d)

        cls = None
        if return_cls:
            cls = self.cls_layers(instance_feature)

        alpha = None
        if self.with_alpha_branch:
            alpha = self.alpha_layers(instance_feature)

        depth = None
        if self.with_depth_branch:
            if self.with_multibin_depth:
                depth = self.depth_layers(instance_feature + anchor2d_embed)
            else:
                focal = torch.cat([
                    metas['focal'][:, i:i+1].repeat(1, qg[1]-qg[0]) for i, qg in enumerate(query_groups)
                ], dim=-1)
                depth = self.depth_layers(instance_feature).exp()
                depth = depth * focal.unsqueeze(-1) / 100

        return output.sigmoid(), cls, depth, alpha