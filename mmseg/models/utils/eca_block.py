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
    """ECA 通道注意力模块.

    与 SE 模块不同，ECA 避免了降维，使用 1D 自适应卷积
    捕获局部跨通道交互，参数量极小（仅 k 个参数，k 通常为 3~7）.

    适用于 SAR 图像场景：自适应增强水体响应通道、抑制相干斑噪声通道.

    Args:
        channels (int): 输入通道数.
        gamma (int): 自适应核大小的 gamma 参数. 默认 2.
        b (int): 自适应核大小的 b 参数. 默认 1.
        init_cfg (dict | None): 初始化配置. 默认 None.
    """

    def __init__(self, channels, gamma=2, b=1, init_cfg=None):
        super().__init__(init_cfg)
        # 自适应核大小: k = |log2(C)/gamma + b/gamma|, 保证奇数
        kernel_size = int(
            abs(
                (torch.log2(
                     torch.tensor(channels, dtype=torch.float)) / gamma +
                 b / gamma).item()))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        kernel_size = max(kernel_size, 3)  # 最小核大小为3

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
            x (Tensor): 输入特征图 (B, C, H, W).

        Returns:
            Tensor: 通道注意力加权后的特征图 (B, C, H, W).
        """
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        y = self.conv(y)  # (B, 1, C)
        y = self.sigmoid(y)  # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)

class SpatialGate(BaseModule):
    """Ultra-lightweight spatial gating for skip connections.

    1. 将 skip 和 gate 特征分别沿通道维求均值 → (B,1,H,W)（零参数）
    2. 相加后经 3×3 depthwise conv 平滑（9 参数）
    3. sigmoid 生成空间注意力图，对 skip 特征逐像素调制
    """

    def __init__(self,
                 skip_channels,
                 gate_channels,
                 use_depthwise_smooth=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.use_depthwise_smooth = use_depthwise_smooth
        if use_depthwise_smooth:
            # 单通道 depthwise 3×3：仅 9 个参数
            self.smooth = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_feat, gate_feat):
        # 通道均值 → (B, 1, H, W)，零参数
        skip_spatial = skip_feat.mean(dim=1, keepdim=True)
        gate_spatial = gate_feat.mean(dim=1, keepdim=True)

        # 门控信号调制跳跃特征
        attention = skip_spatial + gate_spatial

        if self.use_depthwise_smooth:
            attention = self.smooth(attention)  # 9参数平滑

        attention = self.sigmoid(attention)
        return skip_feat * attention  # 空间维逐像素加权


class SkipAttentionGate(BaseModule):
    """跳跃连接注意力门控模块.

    在 concat 之前对编码器跳跃特征施加门控，使用解码器
    上采样后的特征作为 gating signal，自适应过滤噪声.

    适用于 SAR 场景：编码器浅层特征含大量相干斑噪声，
    门控机制可抑制噪声区域、增强水体边界区域的特征传递.

    Args:
        skip_channels (int): 跳跃连接特征通道数（编码器输出）.
        gate_channels (int): 门控信号通道数（解码器上采样后）.
        inter_channels (int | None): 中间层通道数.
            默认 None，取 skip_channels // 2.
        init_cfg (dict | None): 初始化配置. 默认 None.
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

        # 对跳跃特征做 1x1 压缩
        self.W_skip = nn.Sequential(
            nn.Conv2d(
                skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels))
        # 对门控信号做 1x1 压缩
        self.W_gate = nn.Sequential(
            nn.Conv2d(
                gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels))
        # 注意力权重
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, skip_feat, gate_feat):
        """Forward function.

        Args:
            skip_feat (Tensor): 编码器跳跃特征 (B, C_skip, H, W).
            gate_feat (Tensor): 解码器上采样后特征 (B, C_gate, H, W).

        Returns:
            Tensor: 门控后的跳跃特征 (B, C_skip, H, W).
        """
        w_skip = self.W_skip(skip_feat)
        w_gate = self.W_gate(gate_feat)
        # gate_feat 与 skip_feat 空间尺寸相同（已上采样）
        psi = self.psi(nn.ReLU()(w_skip + w_gate))
        return skip_feat * psi

class DSConvBlock(nn.Module):
    """深度可分离卷积块 — Decoder 可选替换

    标准 BasicConvBlock: 2 × (C_in × C_out × 3 × 3) = 18 × C_in × C_out
    DSConvBlock:         2 × (C_in × 9 + C_in × C_out) ≈ 2/9 × BasicConvBlock

    SAR场景：depthwise捕获空间纹理(相干斑/边界)，pointwise做跨通道混合。
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