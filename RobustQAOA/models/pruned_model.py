"""
Pruned CNN construction.

This module physically constructs a smaller neural network when channels are pruned.
It does NOT merely apply a mask — it creates a new Conv2d with fewer output channels
and correctly transfers the surviving weights from the trained baseline.

Critical invariant:
    If conv1 retains k channels (k < 16), the constructed model has:
      - conv1 : Conv2d(3, k, ...)          with k sets of conv1 weights
      - conv2 : Conv2d(k, 32, ...)         with k input channels (rows of conv2.weight trimmed)
      - bn1   : BatchNorm2d(k)             with k parameter sets
    All downstream layers remain unchanged except conv2's in_channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PrunedCNN(nn.Module):
    """
    Physically pruned version of SmallCNN.

    After pruning, the network has fewer parameters and lower memory footprint.
    The forward pass is standard — no masking required.

    Args:
        retained_channels : list of channel indices to keep from the original conv1
        num_classes        : number of output classes (100 for CIFAR-100)
    """

    def __init__(self, retained_channels: List[int], num_classes: int = 100):
        super(PrunedCNN, self).__init__()

        k = len(retained_channels)
        if k == 0:
            raise ValueError("retained_channels must not be empty.")

        self.retained_channels = retained_channels
        self.num_classes       = num_classes
        self.k                 = k

        # Pruned bottleneck: k output channels (down from 16)
        self.conv1 = nn.Conv2d(3, k, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(k)

        # conv2 now receives k channels instead of 16
        self.conv2 = nn.Conv2d(k, 32, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)

        self.fc1     = nn.Linear(64 * 4 * 4, 256)
        self.fc2     = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def build_pruned_model(
    baseline_model: nn.Module,
    retained_channels: List[int],
    num_classes: int = 100,
) -> PrunedCNN:
    """
    Construct a physically pruned model by transferring surviving weights
    from the trained baseline.

    Steps:
        1. Create PrunedCNN with correct channel counts.
        2. Copy conv1 weights for retained output channels only.
        3. Copy bn1 parameters for retained channels only.
        4. Copy conv2 weights for retained input channels only.
        5. Copy all other layers unchanged (conv3, bn2, bn3, fc1, fc2).

    Args:
        baseline_model    : the fully trained SmallCNN
        retained_channels : indices (0-15) of channels to retain
        num_classes        : number of output classes

    Returns:
        PrunedCNN with transferred weights, ready for evaluation.
    """
    pruned = PrunedCNN(retained_channels=retained_channels, num_classes=num_classes)
    idx    = retained_channels                      # shorthand

    with torch.no_grad():
        # ── conv1 weights: shape (16, 3, 3, 3) → (k, 3, 3, 3) ─────────────
        pruned.conv1.weight.copy_(
            baseline_model.conv1.weight[idx, :, :, :]
        )

        # ── bn1: weight, bias, running_mean, running_var ─────────────────────
        pruned.bn1.weight.copy_(baseline_model.bn1.weight[idx])
        pruned.bn1.bias.copy_(baseline_model.bn1.bias[idx])
        pruned.bn1.running_mean.copy_(baseline_model.bn1.running_mean[idx])
        pruned.bn1.running_var.copy_(baseline_model.bn1.running_var[idx])
        pruned.bn1.num_batches_tracked.copy_(baseline_model.bn1.num_batches_tracked)

        # ── conv2 weights: shape (32, 16, 3, 3) → (32, k, 3, 3) ────────────
        pruned.conv2.weight.copy_(
            baseline_model.conv2.weight[:, idx, :, :]
        )

        # ── bn2 (unchanged) ──────────────────────────────────────────────────
        pruned.bn2.weight.copy_(baseline_model.bn2.weight)
        pruned.bn2.bias.copy_(baseline_model.bn2.bias)
        pruned.bn2.running_mean.copy_(baseline_model.bn2.running_mean)
        pruned.bn2.running_var.copy_(baseline_model.bn2.running_var)
        pruned.bn2.num_batches_tracked.copy_(baseline_model.bn2.num_batches_tracked)

        # ── conv3 (unchanged) ────────────────────────────────────────────────
        pruned.conv3.weight.copy_(baseline_model.conv3.weight)

        # ── bn3 (unchanged) ──────────────────────────────────────────────────
        pruned.bn3.weight.copy_(baseline_model.bn3.weight)
        pruned.bn3.bias.copy_(baseline_model.bn3.bias)
        pruned.bn3.running_mean.copy_(baseline_model.bn3.running_mean)
        pruned.bn3.running_var.copy_(baseline_model.bn3.running_var)
        pruned.bn3.num_batches_tracked.copy_(baseline_model.bn3.num_batches_tracked)

        # ── fc1 (unchanged) ──────────────────────────────────────────────────
        pruned.fc1.weight.copy_(baseline_model.fc1.weight)
        pruned.fc1.bias.copy_(baseline_model.fc1.bias)

        # ── fc2 (unchanged) ──────────────────────────────────────────────────
        pruned.fc2.weight.copy_(baseline_model.fc2.weight)
        pruned.fc2.bias.copy_(baseline_model.fc2.bias)

    pruned.eval()
    return pruned
