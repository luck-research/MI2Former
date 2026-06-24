import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn
from ..backbone.mifa.backbone_mifa import MIFABlock
from timm.models.layers import trunc_normal_
from einops import rearrange


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class OctaveBlock(torch.nn.Module):
    def __init__(self, in_dim_h, in_dim_m, in_dim_l, out_dim, kernel_size=3):
        super().__init__()
        # h pass
        self.h2h_conv = Conv2d(in_dim_h, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=in_dim_h)
        
        self.h2m_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.h2m_conv = Conv2d(in_dim_h, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=in_dim_h)

        self.h2l_pool = nn.AvgPool2d(kernel_size=(4, 4), stride=4, padding=0)
        self.h2l_conv = Conv2d(in_dim_h, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=in_dim_h)

        # m pass
        self.m2h_conv = Conv2d(in_dim_m, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=in_dim_m)
        self.m2h_up = nn.Upsample(scale_factor=2, mode="bilinear")

        self.m2m_conv = Conv2d(in_dim_m, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=in_dim_m)

        self.m2l_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.m2l_conv = Conv2d(in_dim_m, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=in_dim_m)

        # l pass
        self.l2h_conv = Conv2d(in_dim_l, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=out_dim)
        self.l2h_up = nn.Upsample(scale_factor=4, mode="bilinear")
        
        self.l2m_conv = Conv2d(in_dim_l, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=out_dim)
        self.l2m_up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        self.l2l_conv = Conv2d(in_dim_l, out_dim, kernel_size=kernel_size, padding=1, bias=False, groups=out_dim)

        # norm
        self.norm_h = nn.GroupNorm(out_dim // 8, out_dim)
        self.norm_m = nn.GroupNorm(out_dim // 8, out_dim)
        self.norm_l = nn.GroupNorm(out_dim // 8, out_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_h, input_m, input_l):
        # High branch
        h_to_h = self.h2h_conv(input_h)

        h_to_m = self.h2m_pool(input_h)
        h_to_m = self.h2m_conv(h_to_m)

        h_to_l = self.h2l_pool(input_h)
        h_to_l = self.h2l_conv(h_to_l)

        # middle branch
        m_to_h = self.m2h_conv(input_m)
        m_to_h = self.m2h_up(m_to_h)

        m_to_m = self.m2m_conv(input_m)

        m_to_l = self.m2l_pool(input_m)
        m_to_l = self.m2l_conv(m_to_l)

        # Low branch
        l_to_h = self.l2h_conv(input_l)
        l_to_h = self.l2h_up(l_to_h)

        l_to_m = self.l2m_conv(input_l)
        l_to_m = self.l2m_up(l_to_m)

        l_to_l = self.l2l_conv(input_l)

        # merge
        h_to_h = h_to_h + m_to_h + l_to_h
        m_to_m = h_to_m + m_to_m + l_to_m
        l_to_l = h_to_l + m_to_l + l_to_l

        # norm
        h_to_h = self.norm_h(h_to_h)
        m_to_m = self.norm_m(m_to_m)
        l_to_l = self.norm_l(l_to_l)

        return h_to_h, m_to_m, l_to_l
    

class MultiscaleOctaveBlock(torch.nn.Module):
    def __init__(self, in_channels, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        self.res3_end3h = OctaveBlock(in_channels[1], in_channels[2], in_channels[3], self.embed_dims)
        self.res4_end3m = OctaveBlock(in_channels[1], in_channels[2], in_channels[3], self.embed_dims)
        self.res5_end3l = OctaveBlock(in_channels[1], in_channels[2], in_channels[3], self.embed_dims)
        self.res3_concat_conv = Conv2d(self.embed_dims * 3, self.embed_dims, kernel_size=1, bias=False, norm=get_norm("GN", self.embed_dims))
        self.res4_concat_conv = Conv2d(self.embed_dims * 3, self.embed_dims, kernel_size=1, bias=False, norm=get_norm("GN", self.embed_dims))
        self.res5_concat_conv = Conv2d(self.embed_dims * 3, self.embed_dims, kernel_size=1, bias=False, norm=get_norm("GN", self.embed_dims))

    def forward(self, res3, res4, res5):
        res3_3, res3_4, res3_5 = self.res3_end3h(res3, res4, res5)
        res4_3, res4_4, res4_5 = self.res4_end3m(res3, res4, res5)
        res5_3, res5_4, res5_5 = self.res5_end3l(res3, res4, res5)

        res3_merge = self.res3_concat_conv(torch.concat([res5_3, res4_3, res3_3], dim=1))
        res4_merge = self.res4_concat_conv(torch.concat([res5_4, res4_4, res3_4], dim=1))
        res5_merge = self.res5_concat_conv(torch.concat([res5_5, res4_5, res3_5], dim=1))
        return res3_merge, res4_merge, res5_merge


class DualBranchAttention(nn.Module):
    def __init__(self, embed_dims, num_heads=8, msda_layers=2):
        super().__init__()

        self.embed_dims = embed_dims
        self.pe_layer = PositionEmbeddingSine(embed_dims // 2, normalize=True)

        self.msda_atten = MSDeformAttnTransformerEncoderOnly(
            d_model=embed_dims,
            dropout=0.1,
            nhead=num_heads,
            dim_feedforward=embed_dims * 4,
            num_encoder_layers=msda_layers,
            num_feature_levels=3,
        )

        self.res3_mifa_atten = MIFABlock(dim=embed_dims, num_heads=num_heads, atten_type='F')
        self.res4_mifa_atten = MIFABlock(dim=embed_dims, num_heads=num_heads, atten_type='F')
        self.res5_mifa_atten = MIFABlock(dim=embed_dims, num_heads=num_heads, atten_type='F')

    def forward(self, res3, res4, res5):
        # msda attention
        pos3 = self.pe_layer(res3)
        pos4 = self.pe_layer(res4)
        pos5 = self.pe_layer(res5)

        res_msda, shapes_list, level_start_index = self.msda_atten([res3, res4, res5], [pos3, pos4, pos5])
        split_section_list = []
        for tmp_shape in shapes_list:
            split_section_list.append(int(tmp_shape[0]) * int(tmp_shape[1]))
        res_msda = torch.split(res_msda, split_section_list, dim=1)

        # mifa attention
        res3_fla = self.res3_mifa_atten(rearrange(res3, 'b c h w -> b (h w) c'), (int(shapes_list[0][0]), int(shapes_list[0][1])))
        res4_fla = self.res4_mifa_atten(rearrange(res4, 'b c h w -> b (h w) c'), (int(shapes_list[1][0]), int(shapes_list[1][1])))
        res5_fla = self.res5_mifa_atten(rearrange(res5, 'b c h w -> b (h w) c'), (int(shapes_list[2][0]), int(shapes_list[2][1])))

        # reshape
        res3_fla = res3_fla.permute(0, 2, 1).reshape(-1, self.embed_dims, int(shapes_list[0][0]), int(shapes_list[0][1])).contiguous()
        res4_fla = res4_fla.permute(0, 2, 1).reshape(-1, self.embed_dims, int(shapes_list[1][0]), int(shapes_list[1][1])).contiguous()
        res5_fla = res5_fla.permute(0, 2, 1).reshape(-1, self.embed_dims, int(shapes_list[2][0]), int(shapes_list[2][1])).contiguous()
                
        res3_msda = res_msda[0].permute(0, 2, 1).reshape(-1, self.embed_dims, int(shapes_list[0][0]), int(shapes_list[0][1])).contiguous()
        res4_msda = res_msda[1].permute(0, 2, 1).reshape(-1, self.embed_dims, int(shapes_list[1][0]), int(shapes_list[1][1])).contiguous()
        res5_msda = res_msda[2].permute(0, 2, 1).reshape(-1, self.embed_dims, int(shapes_list[2][0]), int(shapes_list[2][1])).contiguous()
        
        return [res3_msda, res4_msda, res5_msda], [res3_fla, res4_fla, res5_fla]


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class FocusedLinearAttention(nn.Module):
    def __init__(self, dims, num_heads, focusing_factor=3):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.focusing_factor = focusing_factor
        self.elu = nn.ELU()
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dims)))

    def forward(self, q, k, v, sz2):
        b, head, sz1, _ = v.shape
        assert head == self.num_heads
        c = self.dims // head   
        N = sz1 * sz2

        q = rearrange(q, 'b head sz1 (sz2 c) -> b (sz1 sz2) (head c)', b=b, head=head, sz1=sz1, sz2=sz2, c=c)
        k = rearrange(k, 'b head sz1 (sz2 c) -> b (sz1 sz2) (head c)', b=b, head=head, sz1=sz1, sz2=sz2, c=c)
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q = q / nn.Softplus()(self.scale)
        k = k / nn.Softplus()(self.scale)
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** self.focusing_factor
        k = k ** self.focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = rearrange(q, 'b (sz1 sz2) (head c) -> b head (sz1 sz2) c', b=b, head=head, sz1=sz1, sz2=sz2, c=c)
        k = rearrange(k, 'b (sz1 sz2) (head c) -> b head (sz1 sz2) c', b=b, head=head, sz1=sz1, sz2=sz2, c=c)
        v = rearrange(v, 'b head sz1 (sz2 c) -> b head (sz1 sz2) c', b=b, head=head, sz1=sz1, sz2=sz2, c=c)
 
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1).contiguous() + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        v = q @ kv * z + q

        v = rearrange(v, 'b head (sz1 sz2) c -> b head sz1 (sz2 c)', b=b, head=head, sz1=sz1, sz2=sz2, c=c)
        return v


class CrossInteractionAttention(nn.Module):
    def __init__(self, dims, num_heads):
        super(CrossInteractionAttention, self).__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.project_out1 = nn.Conv2d(dims, dims, kernel_size=1)
        self.project_out2 = nn.Conv2d(dims, dims, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dims, dims, (1, 7), padding=(0, 3), groups=dims)
        self.conv1_1_2 = nn.Conv2d(dims, dims, (1, 11), padding=(0, 5), groups=dims)
        self.conv1_1_3 = nn.Conv2d(dims, dims, (1, 21), padding=(0, 10), groups=dims)
        self.conv1_2_1 = nn.Conv2d(dims, dims, (7, 1), padding=(3, 0), groups=dims)
        self.conv1_2_2 = nn.Conv2d(dims, dims, (11, 1), padding=(5, 0), groups=dims)
        self.conv1_2_3 = nn.Conv2d(dims, dims, (21, 1), padding=(10, 0), groups=dims)

        self.conv2_1_1 = nn.Conv2d(dims, dims, (1, 7), padding=(0, 3), groups=dims)
        self.conv2_1_2 = nn.Conv2d(dims, dims, (1, 11), padding=(0, 5), groups=dims)
        self.conv2_1_3 = nn.Conv2d(dims, dims, (1, 21), padding=(0, 10), groups=dims)
        self.conv2_2_1 = nn.Conv2d(dims, dims, (7, 1), padding=(3, 0), groups=dims)
        self.conv2_2_2 = nn.Conv2d(dims, dims, (11, 1), padding=(5, 0), groups=dims)
        self.conv2_2_3 = nn.Conv2d(dims, dims, (21, 1), padding=(10, 0), groups=dims)

        self.fla1 = FocusedLinearAttention(dims, num_heads)
        self.fla2 = FocusedLinearAttention(dims, num_heads)
        
        self.project_out3 = nn.Conv2d(dims, dims, kernel_size=1)
        self.project_out4 = nn.Conv2d(dims, dims, kernel_size=1)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = to_4d(self.norm1(to_3d(x1)), h, w)
        x2 = to_4d(self.norm2(to_3d(x2)), h, w)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        attn_121 = self.conv1_2_1(x1)
        attn_122 = self.conv1_2_2(x1)
        attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        attn_221 = self.conv2_2_1(x2)
        attn_222 = self.conv2_2_2(x2)
        attn_223 = self.conv2_2_3(x2)

        out1 = attn_111 + attn_112 + attn_113 +attn_121 + attn_122 + attn_123
        out2 = attn_211 + attn_212 + attn_213 +attn_221 + attn_222 + attn_223
        out1 = self.project_out1(out1)
        out2 = self.project_out2(out2)

        # cross focus linear attention
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)

        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        
        out3 = self.fla1(q1, k1, v1, w)
        out4 = self.fla2(q2, k2, v2, h)
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out3(out3) + self.project_out4(out4) + x1 + x2

        return out
    

@SEM_SEG_HEADS_REGISTRY.register()
class HybridCrossScaleAttentionDecoder(nn.Module):
    @configurable
    def __init__(self,
                 in_features: list = ['res2', 'res3', 'res4', 'res5'],
                 in_channels: List[int] = [64, 128, 256, 512],
                 embed_dims: int = 256,
                 mask_dim = 256,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__()
        self.embed_dims = embed_dims

        self.msob = MultiscaleOctaveBlock(in_channels, self.embed_dims)

        self.dba = DualBranchAttention(embed_dims=self.embed_dims, num_heads=8, msda_layers=2)

        self.res3_merge = CrossInteractionAttention(self.embed_dims, 8)
        self.res4_merge = CrossInteractionAttention(self.embed_dims, 8)
        self.res5_merge = CrossInteractionAttention(self.embed_dims, 8)

        # output mask features
        self.res2_lateral_conv = Conv2d(in_channels[0], self.embed_dims, kernel_size=1, bias=False, norm=get_norm("GN", self.embed_dims))
        self.res2_output_conv = Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=1, padding=1, bias=False, norm=get_norm("GN", self.embed_dims), activation=F.leaky_relu)
        self.mask_features = Conv2d(self.embed_dims, mask_dim, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)
        
    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        return ret

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            pass

    @autocast(enabled=False)
    def forward_features(self, features):
        res3, res4, res5 = self.msob(features["res3"], features["res4"], features["res5"])

        res_msda_list, res_fla_list = self.dba(res3, res4, res5)

         # merge
        res3 = self.res3_merge(res_fla_list[0], res_msda_list[0])
        res4 = self.res4_merge(res_fla_list[1], res_msda_list[1])
        res5 = self.res5_merge(res_fla_list[2], res_msda_list[2])

        # out mask
        res2 = self.res2_lateral_conv(features["res2"])
        res2 = res2 + F.interpolate(res3, size=res2.shape[-2:], mode="bilinear", align_corners=False)
        res2 = self.res2_output_conv(res2)
        res2 = self.mask_features(res2)
        return res2, None, [res5, res4, res3]
