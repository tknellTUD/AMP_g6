import torch
import torch.nn as nn
from torchvision.models import resnet18

class ImageBackbone(nn.Module):
    def __init__(self, out_channels=128):
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
        imgs: [B, N, 3, H, W]
        Returns: [B, N, H_out, W_out, C]
        """
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)  # flatten batch and camera dims
        feats = self.encoder(imgs)
        feats = self.output_conv(feats)  # [B*N, C_out, H_feat, W_feat]
        C_out, H_out, W_out = feats.shape[1:]
        feats = feats.view(B, N, C_out, H_out, W_out)
        feats = feats.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C]
        return feats
