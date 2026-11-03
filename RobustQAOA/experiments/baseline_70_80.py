"""
experiments/baseline_70_80.py
==============================
Target 70-80% test accuracy on the fixed 10k/2k/2k CIFAR-100 subset.

Strategy
--------
1. **Architecture** — ResNet18Prunable:
   - Loads ImageNet-pretrained ResNet18 weights.
   - Replaces the original 7×7 / stride-2 stem with a CIFAR-friendly 3×3 / stride-1 stem.
   - Removes the first MaxPool (CIFAR images are 32×32, not 224×224).
   - Inserts a PRUNABLE BOTTLENECK as the very first operation:
       conv1  : Conv2d(3, 16, 3, padding=1, bias=False)   ← 16-channel prunable layer
       bn1    : BatchNorm2d(16)
       proj   : Conv2d(16, 64, 1, bias=False)             ← projects back to ResNet width
       bn_proj: BatchNorm2d(64)
   The rest of the network (layer1–layer4, avgpool, fc) is standard ResNet18
   with fc replaced by Linear(512, 100).
   conv1 stays at exactly 16 output channels → pruning pipeline unchanged.

2. **forward_masked(x, mask)** — applies a (16,) binary/float mask to the
   conv1 output before the projection. Signature is identical to SmallCNN.

3. **Pruning** — build_resnet_pruned_model(model, retained_channels) constructs
   a physically smaller PrunedResNet18 with k<16 channels in conv1/bn1/proj
   input-slice. All downstream ResNet layers are copied unchanged.

4. **Training**:
   - Phase 1 (epochs 1-15) : backbone frozen, only bottleneck + head trained. LR=0.01.
   - Phase 2 (epochs 16-150): full network, SGD+momentum, CosineAnnealingLR. LR=0.1→0.
   - MixUp α=0.4 applied during phase 2.
   - Label smoothing 0.1 throughout.
   - RandAugment (n=2, m=9) as extra training augmentation.
   - Best checkpoint saved by validation accuracy.

5. **No test-set leakage** — test evaluated exactly once after training.

Compatibility contract (must not break)
----------------------------------------
- self.conv1  : Conv2d(3, 16, ...) — name and shape fixed.
- forward_masked(x, mask: tensor(16,)) — identical API to SmallCNN.
- build_resnet_pruned_model(model, retained_channels: List[int]) — same
  contract as build_pruned_model in models/pruned_model.py.
- Checkpoint format: {"state_dict": ..., "metadata": {...}}.
"""

import os
import sys
import json
import time
import random
import math
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as tv_models
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# project imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.model_utils import (
    count_parameters, model_memory_mb,
    measure_inference_time, verify_model_shapes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

OUT_DIR   = "results/baseline_70_80"
CKPT_DIR  = os.path.join(OUT_DIR, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "baseline_model.pt")

# Reuse the same fixed subset indices built in the previous experiment
SUBSET_INDICES_PATH = "results/baseline_small/subset_indices.json"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Architecture
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18Prunable(nn.Module):
    """
    ResNet18-based model with a 16-channel prunable bottleneck.

    Input stem (CIFAR-adapted, no MaxPool):
        conv1    : Conv2d(3,  16, 3, padding=1, bias=False)  ← PRUNABLE BOTTLENECK
        bn1      : BatchNorm2d(16)
        proj     : Conv2d(16, 64, 1, bias=False)             ← 1×1 projection to ResNet width
        bn_proj  : BatchNorm2d(64)
        relu     : ReLU
        (no MaxPool — CIFAR images are already 32×32)

    Backbone: ResNet18 layer1→layer4 (pretrained), avgpool, fc(512→100).

    Memory cost per prunable channel (conv1 output channel i):
        conv1 filter : 3×3×3           = 27 params
        bn1          :                 =  4 params
        proj col     : 64×1×1          = 64 params  (proj input slice)
        subtotal     : 95 params × 4 bytes = 380 bytes/channel
    """

    PRUNABLE_CHANNELS = 16
    BYTES_PER_CHANNEL = 380  # for this architecture (proj replaces wide conv2)

    def __init__(self, num_classes: int = 100, pretrained: bool = True):
        super().__init__()

        # ── PRUNABLE BOTTLENECK ──────────────────────────────────────────
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        # 1×1 projection: 16 → 64 (ResNet18 layer1 expects 64 input channels)
        self.proj    = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(64)

        # ── ResNet18 backbone ────────────────────────────────────────────
        rn = tv_models.resnet18(
            weights=tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Take layer1–layer4, avgpool, and replace fc
        self.layer1  = rn.layer1   # 64  → 64  channels
        self.layer2  = rn.layer2   # 64  → 128 channels
        self.layer3  = rn.layer3   # 128 → 256 channels
        self.layer4  = rn.layer4   # 256 → 512 channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

        # ── Initialise the non-pretrained parts ──────────────────────────
        for m in [self.conv1, self.bn1, self.proj, self.bn_proj, self.fc]:
            self._init_module(m)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias,   0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def _stem(self, x: torch.Tensor) -> torch.Tensor:
        """Prunable stem: conv1 → bn1 → relu → proj → bn_proj → relu."""
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 16, 32, 32)
        x = F.relu(self.bn_proj(self.proj(x))) # (B, 64, 32, 32)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._stem(x)               # (B, 64, 32, 32)
        x = self.layer1(x)              # (B, 64, 32, 32)
        x = self.layer2(x)              # (B,128, 16, 16)
        x = self.layer3(x)              # (B,256,  8,  8)
        x = self.layer4(x)              # (B,512,  4,  4)
        x = self.avgpool(x)             # (B,512,  1,  1)
        x = x.view(x.size(0), -1)       # (B,512)
        x = self.fc(x)                  # (B,100)
        return x

    def forward_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel mask on conv1 outputs.

        Args:
            x    : (B, 3, 32, 32)
            mask : float/bool tensor of shape (16,)
                   mask[i] = 0 disables conv1 channel i

        Returns:
            logits (B, 100)
        """
        # apply mask between conv1 and proj
        x = self.bn1(self.conv1(x))
        x = x * mask.to(x.device).view(1, -1, 1, 1)
        x = F.relu(x)
        x = F.relu(self.bn_proj(self.proj(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def freeze_backbone(self) -> None:
        """Freeze layer1–layer4. Leave stem and fc trainable."""
        for m in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Pruned model for ResNet18Prunable
# ─────────────────────────────────────────────────────────────────────────────

class PrunedResNet18(nn.Module):
    """
    Physically pruned variant of ResNet18Prunable.

    Only conv1/bn1 and the input slice of proj change.
    All downstream ResNet layers are copied unchanged.
    """

    def __init__(self, retained_channels: List[int], num_classes: int = 100):
        super().__init__()
        k = len(retained_channels)
        if k == 0:
            raise ValueError("retained_channels must not be empty.")

        self.retained_channels = retained_channels
        self.k = k

        self.conv1   = nn.Conv2d(3, k, kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(k)
        self.proj    = nn.Conv2d(k, 64, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(64)

        # Downstream layers filled by build_resnet_pruned_model
        self.layer1  = None
        self.layer2  = None
        self.layer3  = None
        self.layer4  = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn_proj(self.proj(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def build_resnet_pruned_model(
    baseline_model: ResNet18Prunable,
    retained_channels: List[int],
    num_classes: int = 100,
) -> PrunedResNet18:
    """
    Construct a physically pruned ResNet18Prunable.

    Weight transfer:
        conv1.weight  : [idx, :, :, :]        (k, 3, 3, 3)
        bn1.*         : indexed by idx
        proj.weight   : [:, idx, :, :]        (64, k, 1, 1)
        bn_proj.*     : copied unchanged (still 64 output channels)
        layer1-4      : deep-copied unchanged
        fc            : copied unchanged

    Args:
        baseline_model    : trained ResNet18Prunable
        retained_channels : list of int indices 0-15
        num_classes       : 100

    Returns:
        PrunedResNet18 in eval() mode with transferred weights.
    """
    import copy
    pruned = PrunedResNet18(retained_channels=retained_channels,
                            num_classes=num_classes)
    idx = retained_channels

    with torch.no_grad():
        # conv1: select retained output channels
        pruned.conv1.weight.copy_(baseline_model.conv1.weight[idx])

        # bn1
        pruned.bn1.weight.copy_(baseline_model.bn1.weight[idx])
        pruned.bn1.bias.copy_(baseline_model.bn1.bias[idx])
        pruned.bn1.running_mean.copy_(baseline_model.bn1.running_mean[idx])
        pruned.bn1.running_var.copy_(baseline_model.bn1.running_var[idx])
        pruned.bn1.num_batches_tracked.copy_(
            baseline_model.bn1.num_batches_tracked)

        # proj: select retained input channels (k input, 64 output unchanged)
        pruned.proj.weight.copy_(
            baseline_model.proj.weight[:, idx, :, :])

        # bn_proj: unchanged (64 channels)
        pruned.bn_proj.load_state_dict(baseline_model.bn_proj.state_dict())

        # backbone layers: deep copy
        pruned.layer1 = copy.deepcopy(baseline_model.layer1)
        pruned.layer2 = copy.deepcopy(baseline_model.layer2)
        pruned.layer3 = copy.deepcopy(baseline_model.layer3)
        pruned.layer4 = copy.deepcopy(baseline_model.layer4)

        # fc
        pruned.fc = copy.deepcopy(baseline_model.fc)

    pruned.eval()
    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _load_subset_indices() -> Tuple[List[int], List[int], List[int]]:
    """Load the fixed subset indices from the previous experiment."""
    if not os.path.exists(SUBSET_INDICES_PATH):
        raise FileNotFoundError(
            f"Subset indices not found at {SUBSET_INDICES_PATH}. "
            "Run train_small_baseline.py first to generate them."
        )
    with open(SUBSET_INDICES_PATH) as f:
        d = json.load(f)
    return d["train_indices"], d["val_indices"], d["test_indices"]


def _verify_no_overlap(train_idx, val_idx) -> None:
    overlap = set(train_idx) & set(val_idx)
    assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} samples"
    print(f"  [subset] Overlap check PASSED — {len(train_idx)} train / "
          f"{len(val_idx)} val / no overlap.")


def get_loaders(
    cfg: Dict[str, Any],
    train_idx: List[int],
    val_idx:   List[int],
    test_idx:  List[int],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build three DataLoaders.

    Training transforms:
        RandAugment(n=2, m=9) + RandomCrop(32, pad=4) + RandomHorizontalFlip
        + ToTensor + Normalize

    Eval transforms:
        ToTensor + Normalize
    """
    data_dir    = cfg.get("data_dir",    "./data")
    batch_size  = cfg.get("batch_size",  64)
    num_workers = cfg.get("num_workers", 0)

    train_transform = transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    raw_train_aug  = torchvision.datasets.CIFAR100(
        root=data_dir, train=True,  download=True, transform=train_transform)
    raw_train_eval = torchvision.datasets.CIFAR100(
        root=data_dir, train=True,  download=True, transform=eval_transform)
    raw_test_eval  = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(
        Subset(raw_train_aug,  train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(raw_train_eval, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(raw_test_eval, test_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MixUp helper
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_loader(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(1)
            correct += pred.eq(targets).sum().item()
            total   += inputs.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    ap = np.array(all_preds)
    at = np.array(all_targets)
    return {
        "loss":      total_loss / total,
        "accuracy":  correct    / total,
        "precision": float(precision_score(at, ap, average="macro", zero_division=0)),
        "recall":    float(recall_score(at,    ap, average="macro", zero_division=0)),
        "f1":        float(f1_score(at,        ap, average="macro", zero_division=0)),
        "n_samples": total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_resnet(
    model:        ResNet18Prunable,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          Dict[str, Any],
    device:       torch.device,
    save_path:    str,
) -> Dict[str, Any]:
    """
    Two-phase training:
      Phase 1: backbone frozen, warm-up the new stem + fc.
      Phase 2: full fine-tuning with MixUp and CosineAnnealingLR.

    Returns history dict.
    """
    total_epochs  = cfg.get("epochs", 150)
    warmup_epochs = cfg.get("warmup_epochs", 15)
    lr_phase1     = cfg.get("lr_phase1", 0.01)
    lr_phase2     = cfg.get("lr_phase2", 0.05)
    momentum      = cfg.get("momentum",  0.9)
    wd            = cfg.get("weight_decay", 5e-4)
    mixup_alpha   = cfg.get("mixup_alpha", 0.4)
    label_smooth  = cfg.get("label_smoothing", 0.1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    criterion_hard   = nn.CrossEntropyLoss()   # for validation/test (no smoothing)

    history: Dict[str, List] = {
        "epoch": [], "phase": [],
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr": [],
    }
    best_val_acc = -1.0
    best_epoch   = -1

    def _make_optimizer(phase: int) -> Tuple[optim.Optimizer, Any]:
        if phase == 1:
            # Only stem + fc + proj are trained
            params = [p for p in model.parameters() if p.requires_grad]
            opt = optim.SGD(params, lr=lr_phase1, momentum=momentum,
                            weight_decay=wd, nesterov=True)
            sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=warmup_epochs, eta_min=lr_phase1 * 0.01
            )
        else:
            opt = optim.SGD(model.parameters(), lr=lr_phase2, momentum=momentum,
                            weight_decay=wd, nesterov=True)
            sched = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_epochs - warmup_epochs, eta_min=lr_phase2 * 1e-3
            )
        return opt, sched

    print(f"\n{'='*68}")
    print("  PHASE 1 — warm-up (backbone frozen)")
    print(f"  Epochs 1–{warmup_epochs}  |  LR={lr_phase1}")
    print(f"{'='*68}")

    # ── Phase 1: backbone frozen ─────────────────────────────────────────
    model.freeze_backbone()
    opt, sched = _make_optimizer(phase=1)

    t_start = time.time()

    for epoch in range(warmup_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        with tqdm(train_loader, desc=f"  WarmUp {epoch+1:3d}/{warmup_epochs}",
                  leave=False) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                opt.zero_grad()
                outputs = model(inputs)
                loss    = criterion_smooth(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                tr_loss    += loss.item() * inputs.size(0)
                pred        = outputs.argmax(1)
                tr_correct += pred.eq(targets).sum().item()
                tr_total   += inputs.size(0)
                pbar.set_postfix(acc=f"{tr_correct/tr_total:.3f}")

        sched.step()
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total
        val_m    = eval_loader(model, val_loader, device, criterion_hard)

        history["epoch"].append(epoch + 1)
        history["phase"].append(1)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_m["accuracy"])
        history["lr"].append(opt.param_groups[0]["lr"])

        print(f"  WU {epoch+1:3d}/{warmup_epochs} | "
              f"tr={tr_acc:.4f} val={val_m['accuracy']:.4f} "
              f"lr={opt.param_groups[0]['lr']:.5f}")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch   = epoch + 1
            _save_ckpt(model, save_path, best_epoch, best_val_acc,
                       val_m["loss"], cfg)

    # ── Phase 2: full fine-tuning ─────────────────────────────────────────
    model.unfreeze_backbone()
    opt, sched = _make_optimizer(phase=2)

    print(f"\n{'='*68}")
    print("  PHASE 2 — full fine-tuning + MixUp")
    print(f"  Epochs {warmup_epochs+1}–{total_epochs}  |  LR={lr_phase2}  "
          f"|  MixUp α={mixup_alpha}")
    print(f"{'='*68}")

    for epoch in range(warmup_epochs, total_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        with tqdm(train_loader,
                  desc=f"  Epoch {epoch+1:4d}/{total_epochs}",
                  leave=False) as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                # MixUp
                inputs_m, y_a, y_b, lam = mixup_data(
                    inputs, targets, mixup_alpha, device
                )
                opt.zero_grad()
                outputs = model(inputs_m)
                loss    = mixup_criterion(criterion_smooth, outputs, y_a, y_b, lam)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                tr_loss += loss.item() * inputs.size(0)
                # For accuracy tracking use non-mixed inputs
                with torch.no_grad():
                    pred = model(inputs).argmax(1)
                tr_correct += pred.eq(targets).sum().item()
                tr_total   += inputs.size(0)
                pbar.set_postfix(acc=f"{tr_correct/tr_total:.3f}")

        sched.step()
        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total
        val_m    = eval_loader(model, val_loader, device, criterion_hard)
        cur_lr   = opt.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["phase"].append(2)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_m["accuracy"])
        history["lr"].append(cur_lr)

        print(f"  E{epoch+1:4d}/{total_epochs} | "
              f"tr={tr_acc:.4f} val={val_m['accuracy']:.4f} "
              f"lr={cur_lr:.6f}")

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch   = epoch + 1
            _save_ckpt(model, save_path, best_epoch, best_val_acc,
                       val_m["loss"], cfg)

    elapsed = time.time() - t_start
    print(f"\n  Training complete: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

    history["best_val_acc"]    = best_val_acc
    history["best_epoch"]      = best_epoch
    history["training_time_s"] = elapsed
    return history


def _save_ckpt(model, path, epoch, val_acc, val_loss, cfg):
    torch.save({
        "state_dict": model.state_dict(),
        "metadata": {
            "epoch":    epoch,
            "val_acc":  val_acc,
            "val_loss": val_loss,
            "config":   cfg,
        },
    }, path)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_plots(history: Dict, out_dir: str) -> None:
    epochs = history["epoch"]

    # Loss
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss", lw=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val loss",   lw=1.5)
    # Phase boundary
    wu = next((e for e, p in zip(history["epoch"], history["phase"]) if p == 2),
              None)
    if wu:
        ax.axvline(wu, color="orange", ls="--", alpha=0.5,
                   label="Phase 2 start")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss (ResNet18 fine-tuned)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, history["train_acc"], label="Train acc", lw=1.5)
    ax.plot(epochs, history["val_acc"],   label="Val acc",   lw=1.5)
    best_ep  = history["best_epoch"]
    best_acc = history["best_val_acc"]
    ax.axvline(best_ep, color="red", ls="--", alpha=0.6,
               label=f"Best val E{best_ep} ({best_acc:.4f})")
    if wu:
        ax.axvline(wu, color="orange", ls="--", alpha=0.5,
                   label="Phase 2 start")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


def _write_outputs(
    out_dir:  str,
    metrics:  Dict,
    history:  Dict,
    cfg:      Dict,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _s(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    # metrics.json
    mc = {k: _s(v) for k, v in metrics.items()}
    mc["training_history"] = {
        k: [float(x) for x in v] if isinstance(v, list) else v
        for k, v in history.items()
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(mc, f, indent=2)

    # metrics.csv
    keys = ["accuracy","precision","recall","f1","loss",
            "parameters","memory_mb","inference_ms","training_time_s",
            "n_train","n_val","n_test"]
    pd.DataFrame([{k: mc[k] for k in keys if k in mc}]).to_csv(
        os.path.join(out_dir, "metrics.csv"), index=False
    )

    # training_history.csv
    if history.get("epoch"):
        pd.DataFrame({
            "epoch":      history["epoch"],
            "phase":      history["phase"],
            "train_loss": history["train_loss"],
            "val_loss":   history["val_loss"],
            "train_acc":  history["train_acc"],
            "val_acc":    history["val_acc"],
            "lr":         history["lr"],
        }).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    # experiment_config.json
    with open(os.path.join(out_dir, "experiment_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # plots
    if history.get("epoch"):
        _save_plots(history, out_dir)

    print(f"  [outputs] All files written to {out_dir}/")


def _generate_diagnostic_report(
    out_dir: str,
    cfg:     Dict,
    metrics: Dict,
    prev_metrics_path: str = "results/baseline_small/metrics.json",
) -> None:
    prev = {}
    if os.path.exists(prev_metrics_path):
        with open(prev_metrics_path) as f:
            prev = json.load(f)

    target = metrics["accuracy"] >= 0.70
    lines = [
        "# Diagnostic Report — ResNet18 Prunable (70-80% Target)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Architecture: ResNet18Prunable",
        "| Component | Description |",
        "|---|---|",
        "| conv1 (PRUNABLE) | Conv2d(3, **16**, 3, padding=1) |",
        "| bn1 | BatchNorm2d(16) |",
        "| proj | Conv2d(16, 64, 1) — projects to ResNet width |",
        "| bn_proj | BatchNorm2d(64) |",
        "| layer1-4 | ResNet18 residual blocks (pretrained) |",
        "| avgpool | AdaptiveAvgPool2d(1,1) |",
        "| fc | Linear(512, 100) |",
        "",
        "## Training Strategy",
        "| Setting | Value |",
        "|---|---|",
        f"| Phase 1 (frozen backbone) | {cfg.get('warmup_epochs',15)} epochs, LR={cfg.get('lr_phase1',0.01)} |",
        f"| Phase 2 (full finetune)   | {cfg.get('epochs',150)-cfg.get('warmup_epochs',15)} epochs, LR={cfg.get('lr_phase2',0.05)} |",
        f"| Scheduler | CosineAnnealingLR per phase |",
        f"| MixUp alpha | {cfg.get('mixup_alpha',0.4)} |",
        f"| Label smoothing | {cfg.get('label_smoothing',0.1)} |",
        f"| RandAugment | n=2, m=9 |",
        f"| Weight decay | {cfg.get('weight_decay',5e-4)} |",
        f"| Pretrained | ImageNet ResNet18 |",
        "",
        "## Dataset",
        f"- Fixed stratified subset (reused from baseline_small experiment)",
        f"- Train: {metrics.get('n_train',10000):,} | Val: {metrics.get('n_val',2000):,} | Test: {metrics.get('n_test',2000):,}",
        "",
        "## Final Test Metrics (fixed 2000-image subset)",
        f"| Test Accuracy    | **{metrics['accuracy']:.4f}** |",
        f"|---|---|",
        f"| Macro Precision  | {metrics['precision']:.4f} |",
        f"| Macro Recall     | {metrics['recall']:.4f} |",
        f"| Macro F1         | {metrics['f1']:.4f} |",
        f"| Test Loss        | {metrics['loss']:.4f} |",
        f"| Parameters       | {metrics['parameters']:,} |",
        f"| Memory (MB)      | {metrics['memory_mb']:.3f} |",
        f"| Inference (ms)   | {metrics['inference_ms']:.4f} |",
        f"| Training Time    | {metrics['training_time_s']:.0f}s ({metrics['training_time_s']/60:.1f} min) |",
        "",
        "## Comparison with Previous Baselines",
        "| Baseline | Test Accuracy | Test F1 | Notes |",
        "|---|---:|---:|---|",
        f"| Original SmallCNN (50 ep, full 50k data) | {prev.get('accuracy', 'N/A') if prev else 'N/A'} | "
        f"{prev.get('f1', 'N/A') if prev else 'N/A'} | 10k test, different split |",
    ]

    prev_small_path = "results/baseline_small/metrics.json"
    if os.path.exists(prev_small_path):
        with open(prev_small_path) as f:
            ps = json.load(f)
        lines.append(
            f"| ImprovedSmallCNN (40 ep, 10k data) | {ps.get('accuracy', 'N/A'):.4f} | "
            f"{ps.get('f1', 'N/A'):.4f} | 2k test subset |"
        )

    lines += [
        f"| **ResNet18Prunable (this run)**       | **{metrics['accuracy']:.4f}** | "
        f"**{metrics['f1']:.4f}** | 2k test subset |",
        "",
        "## 70-80% Target Assessment",
        f"- **Target reached: {'YES' if target else 'NO'}**",
        f"- Achieved: {metrics['accuracy']:.4f}",
        f"- Gap to 70%: {max(0.0, 0.70 - metrics['accuracy']):.4f}",
        "",
        "## Pruning Compatibility",
        "- `conv1` has exactly 16 output channels (verified).",
        "- `forward_masked(x, mask)` applies channel mask before `proj`.",
        "- `build_resnet_pruned_model(model, retained_channels)` transfers weights.",
        "- Downstream ResNet layers are unaffected by pruning.",
        "",
        "## What Improved Accuracy",
        "1. **ImageNet pretraining** — ResNet18 backbone provides rich feature hierarchy.",
        "2. **RandAugment** — stronger on-the-fly augmentation prevents overfitting.",
        "3. **MixUp (alpha=0.4)** — interpolation regularization, especially effective with small data.",
        "4. **Two-phase training** — frozen backbone warm-up prevents catastrophic forgetting.",
        "5. **CosineAnnealingLR** — smooth LR decay, avoids sharp transitions.",
        "",
    ]

    if not target:
        lines += [
            "## Why 70%+ Remains Challenging",
            "- 100 classes from 100 samples/class is severe data scarcity.",
            "- ResNet18 is a 11M-parameter model; overfitting risk is real even with MixUp.",
            "- The 16-channel bottleneck discards information before ResNet can process it.",
            "",
            "## Recommended Next Steps",
            "1. Knowledge distillation from a larger teacher (ResNet50/WideResNet).",
            "2. Test-time augmentation (TTA): average predictions over flipped/cropped copies.",
            "3. Progressive resizing: train at 28px, fine-tune at 32px.",
            "4. AutoAugment policy tuned for CIFAR-100.",
        ]

    with open(os.path.join(out_dir, "diagnostic_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [report] Diagnostic report written.")


def _generate_comparison(out_dir: str, metrics: Dict) -> str:
    baselines = {}
    for name, path in [
        ("Original SmallCNN (50ep, 50k train, 10k test)",
         "results/baseline/metrics.json"),
        ("ImprovedSmallCNN (40ep, 10k train, 2k test)",
         "results/baseline_small/metrics.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                baselines[name] = json.load(f)

    rows = []
    header = ("| Metric | " +
              " | ".join(baselines.keys()) +
              " | **ResNet18Prunable** |")
    sep    = ("|---|" + "---:|" * len(baselines) + "---:|")

    metric_defs = [
        ("Test Accuracy",    "accuracy",      ".4f"),
        ("Macro Precision",  "precision",     ".4f"),
        ("Macro Recall",     "recall",        ".4f"),
        ("Macro F1",         "f1",            ".4f"),
        ("Test Loss",        "loss",          ".4f"),
        ("Parameters",       "parameters",    ",.0f"),
        ("Memory (MB)",      "memory_mb",     ".3f"),
        ("Training Time(s)", "training_time_s",".0f"),
    ]

    md_lines = [
        "# Comparison Table\n",
        "> NOTE: Different test-set sizes make accuracy comparisons approximate.",
        "",
        header, sep,
    ]
    for label, key, fmt in metric_defs:
        vals = []
        for b in baselines.values():
            v = b.get(key)
            vals.append(f"{v:{fmt}}" if v is not None else "N/A")
        nv = metrics.get(key)
        new_str = f"**{nv:{fmt}}**" if nv is not None else "N/A"
        md_lines.append(f"| {label} | " + " | ".join(vals) + f" | {new_str} |")

    md_text = "\n".join(md_lines)
    with open(os.path.join(out_dir, "comparison_report.md"), "w",
              encoding="utf-8") as f:
        f.write(md_text)

    # CSV version
    csv_rows = []
    for label, key, fmt in metric_defs:
        row = {"metric": label}
        for bname, b in baselines.items():
            v = b.get(key)
            row[bname] = f"{v:{fmt}}" if v is not None else "N/A"
        nv = metrics.get(key)
        row["ResNet18Prunable"] = f"{nv:{fmt}}" if nv is not None else "N/A"
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(
        os.path.join(out_dir, "comparison_report.csv"), index=False
    )
    return md_text


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Pruning compatibility verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_pruning_compat(model: ResNet18Prunable, device: torch.device) -> None:
    """Smoke-test the pruning interface."""
    print("\n  [prune check] Verifying pruning compatibility …")

    # a) forward_masked
    dummy = torch.randn(2, 3, 32, 32).to(device)
    mask  = torch.ones(16).to(device)
    out   = model.forward_masked(dummy, mask)
    assert out.shape == (2, 100), f"forward_masked shape wrong: {out.shape}"
    print(f"  [prune check] forward_masked(full mask)   OK — output {tuple(out.shape)}")

    mask_half = mask.clone(); mask_half[:8] = 0
    out2 = model.forward_masked(dummy, mask_half)
    assert out2.shape == (2, 100)
    print(f"  [prune check] forward_masked(half mask)   OK — output {tuple(out2.shape)}")

    # b) build_resnet_pruned_model
    retained = list(range(9))   # keep 9 of 16 channels (56%)
    pruned   = build_resnet_pruned_model(model, retained)
    pruned   = pruned.to(device)
    out3     = pruned(dummy)
    assert out3.shape == (2, 100), f"pruned output shape wrong: {out3.shape}"
    print(f"  [prune check] build_resnet_pruned_model(9) OK — output {tuple(out3.shape)}")

    # c) conv1 shape
    assert model.conv1.weight.shape[0] == 16, "conv1 must have 16 output channels"
    print(f"  [prune check] conv1.weight.shape = {tuple(model.conv1.weight.shape)}  OK")

    print("  [prune check] All checks PASSED.")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG: Dict[str, Any] = {
    "data_dir":        "./data",
    "num_workers":     0,
    "seed":            42,
    "batch_size":      64,
    "epochs":          150,
    "warmup_epochs":   15,
    "lr_phase1":       0.01,
    "lr_phase2":       0.05,
    "momentum":        0.9,
    "weight_decay":    5e-4,
    "mixup_alpha":     0.4,
    "label_smoothing": 0.1,
    "pretrained":      True,
    "force_retrain":   False,
    "out_dir":         OUT_DIR,
    "ckpt_dir":        CKPT_DIR,
}


def run_baseline_70_80(cfg: Optional[Dict[str, Any]] = None) -> Tuple[
    ResNet18Prunable, Dict, DataLoader
]:
    """
    Main experiment.

    Returns: (model, test_metrics, test_loader)
    """
    if cfg is None:
        cfg = dict(DEFAULT_CFG)
    else:
        c = dict(DEFAULT_CFG)
        c.update(cfg)
        cfg = c

    out_dir    = cfg["out_dir"]
    ckpt_dir   = cfg["ckpt_dir"]
    save_path  = os.path.join(ckpt_dir, "baseline_model.pt")
    force_retrain = cfg.get("force_retrain", False)

    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Reproducibility ───────────────────────────────────────────────────
    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"  [seed] {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [device] {device}")

    # ── Load fixed subset indices ─────────────────────────────────────────
    train_idx, val_idx, test_idx = _load_subset_indices()
    _verify_no_overlap(train_idx, val_idx)
    cfg["n_train"] = len(train_idx)
    cfg["n_val"]   = len(val_idx)
    cfg["n_test"]  = len(test_idx)

    # ── Data loaders ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_loaders(
        cfg, train_idx, val_idx, test_idx
    )
    print(f"  [data] {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val / "
          f"{len(test_loader.dataset)} test")

    # ── Build model ───────────────────────────────────────────────────────
    model = ResNet18Prunable(
        num_classes=100, pretrained=cfg.get("pretrained", True)
    ).to(device)
    print(f"  [model] ResNet18Prunable — {count_parameters(model):,} params")

    # ── Train or load ─────────────────────────────────────────────────────
    if os.path.exists(save_path) and not force_retrain:
        print(f"  [ckpt] Loading existing: {save_path}")
        payload = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        meta    = payload.get("metadata", {})
        history = {
            "best_val_acc": meta.get("val_acc", 0),
            "best_epoch":   meta.get("epoch",   0),
            "epoch": [], "phase": [],
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "lr": [], "training_time_s": 0.0,
        }
    else:
        t0      = time.time()
        history = train_resnet(
            model, train_loader, val_loader, cfg, device, save_path
        )
        history["training_time_s"] = time.time() - t0
        # Reload best checkpoint
        payload = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        print("  [ckpt] Best checkpoint reloaded.")

    # ── Shape verification ────────────────────────────────────────────────
    ok = verify_model_shapes(model, device)
    if not ok:
        raise RuntimeError("Model shape check FAILED.")

    # ── Pruning compatibility check ───────────────────────────────────────
    verify_pruning_compat(model, device)

    # ── Test evaluation (ONCE, after training) ────────────────────────────
    model.eval()
    print("\n  [eval] Evaluating on fixed 2k test subset …")
    criterion_hard = nn.CrossEntropyLoss()
    test_m = eval_loader(model, test_loader, device, criterion_hard)

    test_m.update({
        "parameters":       count_parameters(model),
        "memory_mb":        model_memory_mb(model),
        "inference_ms":     measure_inference_time(model, device),
        "training_time_s":  history.get("training_time_s", 0.0),
        "best_val_acc":     history.get("best_val_acc", 0.0),
        "best_epoch":       history.get("best_epoch", 0),
        "device":           str(device),
        "seed":             seed,
        "n_train":          len(train_idx),
        "n_val":            len(val_idx),
        "n_test":           len(test_idx),
        "architecture":     "ResNet18Prunable",
    })

    # ── Save all outputs ──────────────────────────────────────────────────
    _write_outputs(out_dir, test_m, history, cfg)
    _generate_diagnostic_report(out_dir, cfg, test_m)
    cmp_md = _generate_comparison(out_dir, test_m)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  FINAL RESULTS — ResNet18Prunable on fixed 2k test subset")
    print(f"  {'─'*55}")
    print(f"  Test Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  Macro Precision: {test_m['precision']:.4f}")
    print(f"  Macro Recall   : {test_m['recall']:.4f}")
    print(f"  Macro F1       : {test_m['f1']:.4f}")
    print(f"  Test Loss      : {test_m['loss']:.4f}")
    print(f"  Parameters     : {test_m['parameters']:,}")
    print(f"  Memory         : {test_m['memory_mb']:.3f} MB")
    print(f"  Inference      : {test_m['inference_ms']:.4f} ms")
    t_s = test_m['training_time_s']
    print(f"  Training Time  : {t_s:.0f}s ({t_s/60:.1f} min)")
    target_met = test_m['accuracy'] >= 0.70
    print(f"  70% target     : {'REACHED' if target_met else 'NOT REACHED'}")
    print(f"{'='*68}")
    print(f"  Results: {out_dir}/")

    return model, test_m, test_loader
