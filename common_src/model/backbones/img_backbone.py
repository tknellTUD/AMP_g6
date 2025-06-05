import torch
import torch.nn as nn
from torchvision.models import resnet18

class ImageBackbone(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        resnet = resnet18(pretrained=False)
        resnet.load_state_dict(torch.load('common_src/model/middle_encoders/HyDRa/resnet18.pth', map_location='cpu'))
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # downsample x4
            resnet.layer2,  # downsample x8
            resnet.layer3   # downsample x16 
        )
        self.output_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, imgs):
        """
        imgs: [B, H, W, C]
        Returns: [B, N, H_out, W_out, C]
        """
        B, H, W, C = imgs.shape
        imgs = imgs.view(B, C, H, W)  # reshape to [B, H, W, 1, C]
        print(f"imgs shape after reshape: {imgs.shape}")
        feats = self.encoder(imgs)
        feats = self.output_conv(feats)  # [B, C_out, H_feat, W_feat]
        return feats
