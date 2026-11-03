"""
CNN Architecture for CIFAR-100 Channel Pruning Experiments.

Architecture:
  - conv1: Conv2d(3, 16, 3, padding=1)   ← THE PRUNABLE BOTTLENECK (16 output channels)
  - conv2: Conv2d(16, 32, 3, padding=1)
  - conv3: Conv2d(32, 64, 3, padding=1)
  - fc1  : Linear(64*4*4, 256)
  - fc2  : Linear(256, 100)

The layer named 'conv1' with out_channels=16 is the designated prunable layer.
All pruning experiments target this single layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-100 with a clearly identifiable 16-channel prunable bottleneck.

    Input:  (B, 3, 32, 32)
    Output: (B, 100)

    Prunable layer:
        self.conv1  →  Conv2d(3, 16, kernel_size=3, padding=1)
        Identifier: 'conv1'
        Output channels: 16
    """

    def __init__(self, num_classes: int = 100):
        super(SmallCNN, self).__init__()

        # ------------------------------------------------------------------ #
        # PRUNABLE BOTTLENECK LAYER — designated for channel-selection study  #
        # out_channels = 16                                                    #
        # ------------------------------------------------------------------ #
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,       # ← THIS IS THE PRUNABLE LAYER
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        # Downstream layers — receive 16-channel feature maps from conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)

        # After three 2×2 max-pools: 32 → 16 → 8 → 4
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1 — prunable bottleneck
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)          # 32 → 16

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)          # 16 → 8

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)          # 8 → 4

        x = x.view(x.size(0), -1)       # flatten → (B, 1024)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    # ------------------------------------------------------------------ #
    # Masked forward — zero-out selected conv1 output channels            #
    # Used for channel ablation without rebuilding the network             #
    # ------------------------------------------------------------------ #
    def forward_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with a channel mask applied to conv1 outputs.

        Args:
            x    : input tensor  (B, 3, 32, 32)
            mask : boolean/float tensor of shape (16,)
                   mask[i] = 0 disables channel i from conv1

        Returns:
            logits (B, 100)
        """
        x = self.bn1(self.conv1(x))
        # Apply mask: shape (1, 16, 1, 1) for broadcasting
        x = x * mask.to(x.device).view(1, -1, 1, 1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def build_model(num_classes: int = 100) -> SmallCNN:
    """Convenience factory function."""
    return SmallCNN(num_classes=num_classes)
