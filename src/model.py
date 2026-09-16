"""
Convolutional Neural Network architecture for Automated Industrial Defect Detection.
Features multi-scale texture feature extraction blocks, batch normalization,
residual skip connections, and spatial dropout to handle industrial surface variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextureConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(TextureConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return F.relu(out)


class IndustrialDefectCNN(nn.Module):
    """
    Surface defect classification network tailored for NEU 6-class steel inspection.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super(IndustrialDefectCNN, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False), # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64 -> 32
        )

        self.stage1 = TextureConvBlock(32, 64, stride=1)   # 32x32
        self.stage2 = TextureConvBlock(64, 128, stride=2)  # 16x16
        self.stage3 = TextureConvBlock(128, 256, stride=2) # 8x8

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        out = self.classifier(flat)
        return out
