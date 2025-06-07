import os
from torch import nn
from torchvision.models import resnet50
import torch

class ResNetBackbone(nn.Module):
    """ResNet backbone adapted for LiDAR data and custom output channels."""
    def __init__(self, in_channels=64, weights_path=None):
        super().__init__()
        # Load base ResNet without weights (so we can custom-load them)
        self.resnet = resnet50(weights=None)

        # Replace first conv layer to match LiDAR feature input
        self.resnet.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Define three blocks
        self.block1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1  # Output: 256 channels
        )
        self.block2 = self.resnet.layer2  # Output: 512 channels
        self.block3 = self.resnet.layer3  # Output: 1024 channels

        # Project to desired output channels: [64, 128, 256]
        self.out_proj1 = nn.Conv2d(256, 64, kernel_size=1)
        self.out_proj2 = nn.Conv2d(512, 128, kernel_size=1)
        self.out_proj3 = nn.Conv2d(1024, 256, kernel_size=1)

        # Load pretrained weights (with strict=False to ignore shape mismatches)
        if weights_path is not None:
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, weights_only=False)
                self.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {weights_path}")
            else:
                raise FileNotFoundError(f"weights_path does not exist: {weights_path}")
        else:
            print("No custom weights provided, using default ResNet initialization.")

    def forward(self, x):
        """Forward pass through the backbone."""
        outs = []

        x = self.block1(x)       # [B, 256, H/4, W/4]
        x1 = self.out_proj1(x)   # -> [B, 64, H/4, W/4]
        outs.append(x1)

        x = self.block2(x)       # [B, 512, H/8, W/8]
        x2 = self.out_proj2(x)   # -> [B, 128, H/8, W/8]
        outs.append(x2)

        x = self.block3(x)       # [B, 1024, H/16, W/16]
        x3 = self.out_proj3(x)   # -> [B, 256, H/16, W/16]
        outs.append(x3)

        return outs  # List of 3 outputs with target channels

# Example test
if __name__ == "__main__":
    backbone_config = {
        'in_channels': 64,
        'weights_path': '/home/rwsteen/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
    }
    model = ResNetBackbone(**backbone_config)
    dummy_input = torch.randn(4, 64, 320, 320)
    output = model(dummy_input)

    for i, o in enumerate(output):
        print(f"Output {i+1} shape: {o.shape}")
