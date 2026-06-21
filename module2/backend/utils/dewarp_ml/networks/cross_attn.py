"""
Cross-attention encoder/decoder — adapted for dewarp_ml.
Original: Copyright (c) OpenMMLab.
Adaptation: Removed mmcv.runner.BaseModule and mmcv.cnn.ConvModule;
            replaced with pure nn.Module and nn.Sequential equivalents.
"""

import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# mmcv.cnn.ConvModule replacement — pure PyTorch
# ---------------------------------------------------------------------------

def _conv_module(in_channels, out_channels, kernel_size, padding=0,
                 bias=False, groups=1):
    """
    Drop-in for mmcv's ConvModule with BN + ReLU.
    Uses OrderedDict to match mmcv's exact layer naming:
      'conv'     -> Conv2d
      'bn'       -> BatchNorm2d
      'activate' -> ReLU
    This ensures state_dict keys match the 30.pt checkpoint exactly.
    """
    return nn.Sequential(OrderedDict([
        ('conv',     nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=padding, bias=bias, groups=groups)),
        ('bn',       nn.BatchNorm2d(out_channels)),
        ('activate', nn.ReLU(inplace=True)),
    ]))


# ---------------------------------------------------------------------------
# Feedforward
# ---------------------------------------------------------------------------

class LocalityAwareFeedforward(nn.Module):
    """Locality-aware feedforward layer (SATRN style), mmcv-free."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.conv1 = _conv_module(d_in, d_hid, kernel_size=1, padding=0)
        self.depthwise_conv = _conv_module(d_hid, d_hid, kernel_size=3,
                                           padding=1, groups=d_hid)
        self.conv2 = _conv_module(d_hid, d_in, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
# Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


# ---------------------------------------------------------------------------
# Adaptive 2D Positional Encoding  (was BaseModule — now plain nn.Module)
# ---------------------------------------------------------------------------

class Adaptive2DPositionalEncoding(nn.Module):
    """
    Adaptive 2D positional encoder (SATRN style).
    Replaced mmcv BaseModule with plain nn.Module.
    """

    def __init__(self, d_hid=512, n_height=100, n_width=100, dropout=0.1):
        super().__init__()

        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        h_position_encoder = h_position_encoder.transpose(0, 1).view(1, d_hid, n_height, 1)

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = w_position_encoder.transpose(0, 1).view(1, d_hid, 1, n_width)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self._scale_factor_generate(d_hid)
        self.w_scale = self._scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table

    def _scale_factor_generate(self, d_hid):
        return nn.Sequential(
            nn.Conv2d(d_hid, d_hid, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_hid, d_hid, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.pool(x)
        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]
        out = x + h_pos_encoding + w_pos_encoding
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, d_k=64, d_v=64,
                 dropout=0.1, qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.attention = ScaledDotProductAttention(d_k ** 0.5, dropout)
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, len_q, self.dim_v)
        attn_out = self.fc(attn_out)
        return self.proj_drop(attn_out)


# ---------------------------------------------------------------------------
# Cross-Attention Layer
# ---------------------------------------------------------------------------

class CrossattnLayer(nn.Module):
    def __init__(self, d_model=512, d_inner=512, n_head=8, d_k=64, d_v=64,
                 dropout=0.1, qkv_bias=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                       qkv_bias=qkv_bias, dropout=dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                             qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.feed_forward = LocalityAwareFeedforward(d_model, d_inner, dropout=dropout)

    def forward(self, x, cross, h, w, mask=None):
        n, hw, c = x.size()
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(cross, x, x, mask)
        residual = x
        x = self.norm3(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        x = self.feed_forward(x)
        x = x.view(n, c, hw).transpose(1, 2)
        return residual + x


# ---------------------------------------------------------------------------
# CrossEncoder
# ---------------------------------------------------------------------------

class CrossEncoder(nn.Module):
    def __init__(self, n_layers=12, n_head=8, d_k=64, d_v=64,
                 d_model=512, n_position=100, d_inner=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model, n_height=n_position, n_width=n_position, dropout=dropout)
        self.position_enc_cross = Adaptive2DPositionalEncoding(
            d_hid=d_model, n_height=n_position, n_width=n_position, dropout=dropout)
        self.layer_stack = nn.ModuleList([
            CrossattnLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, cross_feat, img_metas=None):
        valid_ratios = [1.0 for _ in range(feat.size(0))]
        feat = self.position_enc(feat)
        cross_feat = self.position_enc_cross(cross_feat)

        n, c, h, w = feat.size()
        mask = feat.new_zeros((n, h, w))
        for i, valid_ratio in enumerate(valid_ratios):
            valid_width = min(w, math.ceil(w * valid_ratio))
            mask[i, :, :valid_width] = 1
        mask = mask.view(n, h * w)
        feat = feat.view(n, c, h * w)
        cross_feat = cross_feat.view(n, c, h * w)

        output = feat.permute(0, 2, 1).contiguous()
        cross = cross_feat.permute(0, 2, 1).contiguous()
        for enc_layer in self.layer_stack:
            output = enc_layer(output, cross, h, w, mask)
        return self.layer_norm(output)


# ---------------------------------------------------------------------------
# Decoder Layer + Decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, d_inner=256, n_head=8, d_k=64, d_v=64,
                 n_position=100, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_head, d_model, d_k, d_v,
                                       qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = LocalityAwareFeedforward(d_model, d_inner, dropout=dropout)

    def forward(self, x, h, w, mask=None):
        n, hw, c = x.size()
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        x = self.feed_forward(x)
        x = x.view(n, c, hw).transpose(1, 2)
        return residual + x


class Decoder(nn.Module):
    def __init__(self, n_layers=4, n_head=8, d_k=64, d_v=64,
                 d_model=512, n_position=100, d_inner=256,
                 dropout=0.1, qkv_bias=False):
        super().__init__()
        self.d_model = d_model
        self.position_dec = Adaptive2DPositionalEncoding(
            d_hid=d_model, n_height=n_position, n_width=n_position, dropout=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat):
        feat = self.position_dec(feat)
        n, c, h, w = feat.size()
        mask = feat.new_zeros((n, h, w))
        for i in range(n):
            mask[i, :, :] = 1
        mask = mask.view(n, h * w)
        feat = feat.view(n, c, h * w)
        output = feat.permute(0, 2, 1).contiguous()
        for dec_layer in self.layer_stack:
            output = dec_layer(output, h, w, mask)
        return self.layer_norm(output)
