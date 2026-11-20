# Copyright (c) OpenMMLab. All rights reserved.
"""Efficient Channel Attention (ECA) and Skip Attention Gate modules.

References:
    - ECA-Net: Efficient Channel Attention for Deep Convolutional Neural
      Networks (CVPR 2020).
    - Attention U-Net: Learning Where to Look for the Pancreas (MICCAI 2018).
"""
import torch
import torch.nn as nn
from mmengine.model import BaseModule


class ECABlock(BaseModule):
    """Efficient Channel Attention (ECA) block.

    Unlike squeeze-and-excitation, ECA avoids dimensionality reduction and
    captures local cross-channel interaction with a 1D adaptive convolution,
    so it adds only k parameters (k is typically between 3 and 7).

    Args:
        channels (int): Number of input channels.
        gamma (int): Gamma parameter of the adaptive kernel size. Default: 2.
        b (int): b parameter of the adaptive kernel size. Default: 1.
        init_cfg (dict | None): Initialization config. Default: None.
    """

    def __init__(self, channels, gamma=2, b=1, init_cfg=None):
        super().__init__(init_cfg)
        # Adaptive kernel size: k = |log2(C) / gamma + b / gamma|, forced odd
        kernel_size = int(
            abs(
                (torch.log2(
                     torch.tensor(channels, dtype=torch.float)) / gamma +
                 b / gamma).item()))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 3)  # the kernel is at least 3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            Tensor: Channel-wise re-weighted feature map (B, C, H, W).
        """
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)

class SpatialGate(BaseModule):
    """Ultra-lightweight spatial gating for skip connections.

    1. Average the skip and gate features over the channel axis to obtain
       (B, 1, H, W) maps, which costs no parameters.
    2. Sum the two maps and smooth the sum with a 3 x 3 depthwise convolution
       (9 parameters).
    3. Apply a sigmoid to form a spatial attention map and modulate the skip
       features pixel by pixel.
    """

    def __init__(self,
                 skip_channels,
                 gate_channels,
                 use_depthwise_smooth=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.use_depthwise_smooth = use_depthwise_smooth
        if use_depthwise_smooth:
            # single-channel 3 x 3 depthwise convolution: 9 parameters
            self.smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_feat, gate_feat):
        # channel means of shape (B, 1, H, W), no parameters
        skip_spatial = skip_feat.mean(dim=1, keepdim=True)
        gate_spatial = gate_feat.mean(dim=1, keepdim=True)

        # the gate signal modulates the skip features
        attention = skip_spatial + gate_spatial

        if self.use_depthwise_smooth:
            attention = self.smooth(attention)  # 9-parameter smoothing

        attention = self.sigmoid(attention)
        return skip_feat * attention  # pixel-wise weighting


class SkipAttentionGate(BaseModule):
    """Attention gate on the skip connection.

    The gate is applied to the encoder skip features before concatenation,
    with the upsampled decoder features as the gating signal, so that noise
    regions are suppressed adaptively.

    Args:
        skip_channels (int): Number of channels of the skip features.
        gate_channels (int): Number of channels of the gating signal.
        inter_channels (int | None): Number of intermediate channels.
            Default: None, which sets it to skip_channels // 2.
        init_cfg (dict | None): Initialization config. Default: None.
    """

    def __init__(self,
                 skip_channels,
                 gate_channels,
                 inter_channels=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        inter_channels = inter_channels or skip_channels // 2
        if inter_channels == 0:
            inter_channels = 1

        # 1 x 1 compression of the skip features
        self.W_skip = nn.Sequential(
            nn.Conv2d(
                skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels))
        # 1 x 1 compression of the gating signal
        self.W_gate = nn.Sequential(
            nn.Conv2d(
                gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels))
        # attention weights
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, skip_feat, gate_feat):
        """Forward function.

        Args:
            skip_feat (Tensor): Encoder skip features (B, C_skip, H, W).
            gate_feat (Tensor): Upsampled decoder features (B, C_gate, H, W).

        Returns:
            Tensor: Gated skip features (B, C_skip, H, W).
        """
        w_skip = self.W_skip(skip_feat)
        w_gate = self.W_gate(gate_feat)
        # gate_feat and skip_feat have the same spatial size
        psi = self.psi(nn.ReLU()(w_skip + w_gate))
        return skip_feat * psi

class DSConvBlock(nn.Module):
    """Depthwise-separable convolution block.

    A standard BasicConvBlock costs 2 x (C_in x C_out x 3 x 3), that is,
    18 C_in C_out parameters, whereas this block costs 2 x (9 C_in + C_in C_out),
    about 2/9 of the standard block. The depthwise convolution captures spatial
    texture and boundaries, and the pointwise convolution mixes channels.
    """

    def __init__(self, in_channels, out_channels, num_convs=2,
                 stride=1, dilation=1, with_cp=False,
                 conv_cfg=None, norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'), dcn=None, plugins=None):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        self.with_cp = with_cp

        convs = []
        for i in range(num_convs):
            convs.append(
                DepthwiseSeparableConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation if i == 0 else dilation,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = torch.utils.checkpoint.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out