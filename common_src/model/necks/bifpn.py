import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class WeightedAdd(nn.Module):
    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32), requires_grad=True)
    def forward(self, inputs):
        w = F.relu(self.weights)
        norm_w = w / (torch.sum(w) + self.eps)
        out = 0
        for i in range(len(inputs)):
            out += norm_w[i] * inputs[i]
        return out

class BiFPNBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv_p3 = ConvBNReLU(out_channels, out_channels)
        self.conv_p4 = ConvBNReLU(out_channels, out_channels)
        self.conv_p5 = ConvBNReLU(out_channels, out_channels)
        self.wadd_topdown1 = WeightedAdd(2)
        self.wadd_topdown2 = WeightedAdd(2)
        self.wadd_bottomup1 = WeightedAdd(2)
        self.wadd_bottomup2 = WeightedAdd(2)

    def forward(self, features):
        P3, P4, P5 = features
        # Top-down path
        P5_td = P5
        P4_td = self.wadd_topdown2([P4, F.interpolate(P5_td, size=P4.shape[-2:], mode='nearest')])
        P3_td = self.wadd_topdown1([P3, F.interpolate(P4_td, size=P3.shape[-2:], mode='nearest')])
        P3_out = self.conv_p3(P3_td)
        # Bottom-up path
        P4_bu = self.wadd_bottomup1([P4_td, F.max_pool2d(P3_out, kernel_size=2)])
        P4_out = self.conv_p4(P4_bu)
        P5_bu = self.wadd_bottomup2([P5_td, F.max_pool2d(P4_out, kernel_size=2)])
        P5_out = self.conv_p5(P5_bu)
        return [P3_out, P4_out, P5_out]

class BiFPN(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], out_channels=[256, 256, 256], num_layers=2):
        super().__init__()
        assert len(in_channels) == len(out_channels) == 3
        self.input_proj = nn.ModuleList([
            ConvBNReLU(in_c, out_c)
            for in_c, out_c in zip(in_channels, out_channels)
        ])
        self.bifpn_blocks = nn.Sequential(*[BiFPNBlock(out_channels[0]) for _ in range(num_layers)])

    def forward(self, inputs):
        feats = [proj(inp) for proj, inp in zip(self.input_proj, inputs)]  # [B, 256, H, W] (varied H, W)
        feats = self.bifpn_blocks(feats)  # still list of 3 [B, 256, H, W]
        # --- Upsample all to highest resolution (P3) ---
        target_size = feats[0].shape[-2:]
        feats_upsampled = [F.interpolate(f, size=target_size, mode='nearest') for f in feats]
        cat_fused = torch.cat(feats_upsampled, dim=1)  # [B, 768, H, W]
        return [cat_fused]   # <-- match SECONDFPN style!
