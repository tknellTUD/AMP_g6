import torch
import torch.nn as nn
import torch.nn.functional as F

class LidarDepthRefiner(nn.Module):
    def __init__(self, ldc_config):
        bev_channels = ldc_config['bev_channels']
        super().__init__()
        self.conv = nn.Conv2d(bev_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bev_feat, img_feat_bev):
        """
        bev_feat: [B, C, H, W] from LiDAR
        img_feat_bev: [B, C, H, W] from camera
        """
        attn = self.sigmoid(self.conv(bev_feat))
        refined = img_feat_bev * attn + bev_feat * (1 - attn)
        return refined
