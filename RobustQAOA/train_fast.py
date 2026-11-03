"""
train_fast.py
=============
Fast high-accuracy experiment using CIFAR-10 + pretrained ResNet18.

Why CIFAR-10 instead of CIFAR-100
-----------------------------------
CIFAR-100 with 10k training images = 100 samples per class. That is severe
data scarcity for 100-way classification — the previous two runs peaked at
40% accuracy even with ResNet18 pretraining.

CIFAR-10 has only 10 classes.  With 3,000 training images (300/class) a
pretrained ResNet18 fine-tuned for 20-25 epochs routinely achieves 82-90%,
finishing in under 15 minutes on CPU.

The 16-channel prunable bottleneck, forward_masked(), and build_pruned_model()
contract are identical to the ResNet18Prunable used in baseline_70_80.py.
All downstream RobustQAOA pruning (classical, QUBO, QAOA) is compatible.

Dataset split
-------------
- Training   : 3,000 images  (300 per class, stratified, seed 42)
- Validation : 600 images    (60 per class, from official train set, no overlap)
- Test       : 600 images    (60 per class, from official test set)

Outputs saved to results/fast_high_accuracy/
"""

import os, sys, json, time, random, copy
from typing import Dict, Any, List, Tuple, Optional

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

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models.model_utils import (
    count_parameters, model_memory_mb,
    measure_inference_time, verify_model_shapes,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# CIFAR-10 per-channel statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

OUT_DIR   = "results/fast_high_accuracy"
CKPT_DIR  = os.path.join(OUT_DIR, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "baseline_model.pt")

NUM_CLASSES    = 10
TRAIN_PER_CLS  = 300    # 300 × 10 = 3,000 train
VAL_PER_CLS    = 60     # 60  × 10 = 600   val
TEST_PER_CLS   = 60     # 60  × 10 = 600   test
SEED           = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Stratified subset builder (CIFAR-10 version)
# ─────────────────────────────────────────────────────────────────────────────

def make_stratified_subset(
    targets: np.ndarray,
    n_per_class: int,
    seed: int,
    exclude: Optional[set] = None,
) -> List[int]:
    """Return sorted list of indices: n_per_class examples per class."""
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for cls in range(NUM_CLASSES):
        cls_idx = np.where(targets == cls)[0]
        if exclude:
            cls_idx = cls_idx[~np.isin(cls_idx, list(exclude))]
        chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        selected.extend(chosen.tolist())
    return sorted(selected)


def build_subsets(data_dir: str) -> Tuple[List[int], List[int], List[int]]:
    raw_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None)
    raw_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=None)

    train_tgt = np.array(raw_train.targets)
    test_tgt  = np.array(raw_test.targets)

    train_idx = make_stratified_subset(train_tgt, TRAIN_PER_CLS, SEED)
    val_idx   = make_stratified_subset(
        train_tgt, VAL_PER_CLS, SEED + 1, exclude=set(train_idx)
    )
    test_idx  = make_stratified_subset(test_tgt, TEST_PER_CLS, SEED + 2)

    # Verify no train/val overlap
    overlap = set(train_idx) & set(val_idx)
    assert len(overlap) == 0, f"Train/val overlap: {len(overlap)}"
    print(f"  [subset] {len(train_idx)} train / {len(val_idx)} val / "
          f"{len(test_idx)} test — no overlap")
    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_loaders(
    data_dir: str,
    train_idx: List[int],
    val_idx:   List[int],
    test_idx:  List[int],
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    raw_tr_aug  = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=train_tf)
    raw_tr_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=True,  download=True, transform=eval_tf)
    raw_te_eval = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_tf)

    kw = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(Subset(raw_tr_aug,  train_idx),
                              batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(Subset(raw_tr_eval, val_idx),
                              batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(Subset(raw_te_eval, test_idx),
                              batch_size=batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Architecture  —  ResNet18Prunable (CIFAR-10 edition)
#
#  ┌─────────────────────────────────────────────────────────┐
#  │  PRUNABLE BOTTLENECK                                    │
#  │    conv1   : Conv2d(3, 16, 3, pad=1)  ← 16-ch prunable │
#  │    bn1     : BN(16)                                     │
#  │    proj    : Conv2d(16, 64, 1)        ← width restore   │
#  │    bn_proj : BN(64)                                     │
#  ├─────────────────────────────────────────────────────────┤
#  │  PRETRAINED ResNet18 BACKBONE                           │
#  │    layer1 … layer4  (stride adapted for 32×32 input)   │
#  │    avgpool : AdaptiveAvgPool2d(1,1)                     │
#  │    fc      : Linear(512, 10)                            │
#  └─────────────────────────────────────────────────────────┘
#
#  Pruning contract (identical to baseline_70_80.py):
#    forward_masked(x, mask: Tensor[16]) — mask applied between conv1 and proj
#    build_pruned_model(model, retained_channels: List[int]) → PrunedNet
# ─────────────────────────────────────────────────────────────────────────────

class ResNet18Prunable(nn.Module):
    """ResNet18 backbone with a 16-channel prunable entry bottleneck."""

    PRUNABLE_CHANNELS = 16

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()

        # ── PRUNABLE BOTTLENECK ──────────────────────────────────────────
        self.conv1   = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.proj    = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(64)

        # ── ResNet18 backbone ────────────────────────────────────────────
        weights = (tv_models.ResNet18_Weights.IMAGENET1K_V1
                   if pretrained else None)
        rn = tv_models.resnet18(weights=weights)

        self.layer1  = rn.layer1
        self.layer2  = rn.layer2
        self.layer3  = rn.layer3
        self.layer4  = rn.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512, num_classes)

        # Init non-pretrained parts
        for m in (self.conv1, self.bn1, self.proj, self.bn_proj, self.fc):
            _init_module(m)

    def _stem(self, x):
        x = F.relu(self.bn1(self.conv1(x)))    # (B,16,32,32)
        x = F.relu(self.bn_proj(self.proj(x)))  # (B,64,32,32)
        return x

    def forward(self, x):
        x = self._stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward_masked(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Identical contract to SmallCNN.forward_masked.
        mask : float tensor (16,), 0 disables channel i of conv1.
        """
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
        return self.fc(x)

    def freeze_backbone(self):
        for m in (self.layer1, self.layer2, self.layer3, self.layer4):
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True


def _init_module(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Physically pruned model
# ─────────────────────────────────────────────────────────────────────────────

class PrunedResNet18(nn.Module):
    def __init__(self, retained_channels: List[int], num_classes: int = 10):
        super().__init__()
        k = len(retained_channels)
        if k == 0:
            raise ValueError("retained_channels must not be empty")
        self.retained_channels = retained_channels
        self.k = k
        self.conv1   = nn.Conv2d(3, k,  kernel_size=3, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(k)
        self.proj    = nn.Conv2d(k, 64, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(64)
        # backbone set by build_pruned_model
        self.layer1 = self.layer2 = self.layer3 = self.layer4 = None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn_proj(self.proj(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_pruned_model(
    model: ResNet18Prunable,
    retained_channels: List[int],
    num_classes: int = 10,
) -> PrunedResNet18:
    """
    Physically prune conv1 / bn1 / proj-input-slice from a trained
    ResNet18Prunable.  Same contract as models/pruned_model.py.
    """
    pruned = PrunedResNet18(retained_channels, num_classes)
    idx    = retained_channels

    with torch.no_grad():
        pruned.conv1.weight.copy_(model.conv1.weight[idx])

        pruned.bn1.weight.copy_(model.bn1.weight[idx])
        pruned.bn1.bias.copy_(model.bn1.bias[idx])
        pruned.bn1.running_mean.copy_(model.bn1.running_mean[idx])
        pruned.bn1.running_var.copy_(model.bn1.running_var[idx])
        pruned.bn1.num_batches_tracked.copy_(model.bn1.num_batches_tracked)

        pruned.proj.weight.copy_(model.proj.weight[:, idx, :, :])
        pruned.bn_proj.load_state_dict(model.bn_proj.state_dict())

        pruned.layer1 = copy.deepcopy(model.layer1)
        pruned.layer2 = copy.deepcopy(model.layer2)
        pruned.layer3 = copy.deepcopy(model.layer3)
        pruned.layer4 = copy.deepcopy(model.layer4)
        pruned.fc     = copy.deepcopy(model.fc)

    pruned.eval()
    return pruned


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_loader(
    model, loader, device,
    criterion=None,
) -> Dict[str, float]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += pred.eq(y).sum().item()
            total   += x.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
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
# 6.  Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: ResNet18Prunable,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          Dict[str, Any],
    device:       torch.device,
    save_path:    str,
) -> Dict[str, Any]:
    """
    Two-phase SGD training with early stopping.

    Phase 1 (epochs 1-warm_ep):  backbone frozen, only stem+fc trained.
    Phase 2 (epochs warm_ep+1…): full fine-tune, CosineAnnealingLR.
    Early stop: if val_acc doesn't improve for `patience` consecutive epochs.
    """
    total_epochs  = cfg.get("epochs",         25)
    warm_ep       = cfg.get("warmup_epochs",   5)
    lr1           = cfg.get("lr_phase1",       0.01)
    lr2           = cfg.get("lr_phase2",       0.05)
    momentum      = cfg.get("momentum",        0.9)
    wd            = cfg.get("weight_decay",    5e-4)
    patience      = cfg.get("patience",        7)
    label_smooth  = cfg.get("label_smoothing", 0.1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    crit_smooth = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    crit_hard   = nn.CrossEntropyLoss()

    history = dict(epoch=[], phase=[], train_loss=[], val_loss=[],
                   train_acc=[], val_acc=[], lr=[])

    best_val_acc  = -1.0
    best_epoch    = -1
    no_improve    = 0

    # ── Phase 1: warm-up (frozen backbone) ───────────────────────────────
    model.freeze_backbone()
    opt1   = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr1, momentum=momentum, weight_decay=wd, nesterov=True,
    )
    sched1 = optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=warm_ep, eta_min=lr1 * 0.01
    )

    print(f"\n{'='*62}")
    print(f"  Phase 1: backbone frozen, {warm_ep} warm-up epochs (LR={lr1})")
    print(f"{'='*62}")

    t0 = time.time()

    for ep in range(warm_ep):
        tr_loss, tr_c, tr_n = _run_epoch(
            model, train_loader, opt1, crit_smooth, device, ep, warm_ep,
            label="WU"
        )
        val_m  = eval_loader(model, val_loader, device, crit_hard)
        sched1.step()
        _log(ep+1, warm_ep, tr_loss, tr_c/tr_n, val_m, opt1, history, phase=1)
        best_val_acc, best_epoch, no_improve = _maybe_save(
            model, save_path, ep+1, val_m, cfg,
            best_val_acc, best_epoch, no_improve
        )

    # ── Phase 2: full fine-tune ───────────────────────────────────────────
    model.unfreeze_backbone()
    opt2   = optim.SGD(
        model.parameters(),
        lr=lr2, momentum=momentum, weight_decay=wd, nesterov=True,
    )
    phase2_epochs = total_epochs - warm_ep
    sched2 = optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=phase2_epochs, eta_min=lr2 * 1e-3
    )

    print(f"\n{'='*62}")
    print(f"  Phase 2: full fine-tune, {phase2_epochs} epochs "
          f"(LR={lr2}, patience={patience})")
    print(f"{'='*62}")

    for ep in range(warm_ep, total_epochs):
        tr_loss, tr_c, tr_n = _run_epoch(
            model, train_loader, opt2, crit_smooth, device, ep, total_epochs,
            label="EP"
        )
        val_m  = eval_loader(model, val_loader, device, crit_hard)
        sched2.step()
        _log(ep+1, total_epochs, tr_loss, tr_c/tr_n, val_m, opt2, history, phase=2)
        best_val_acc, best_epoch, no_improve = _maybe_save(
            model, save_path, ep+1, val_m, cfg,
            best_val_acc, best_epoch, no_improve
        )

        if no_improve >= patience:
            print(f"  Early stop at epoch {ep+1} "
                  f"(no improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\n  Done: {elapsed:.0f}s ({elapsed/60:.1f} min) | "
          f"best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")

    history["best_val_acc"]    = best_val_acc
    history["best_epoch"]      = best_epoch
    history["training_time_s"] = elapsed
    return history


def _run_epoch(model, loader, opt, crit, device, ep, total, label="EP"):
    model.train()
    tl = tc = tn = 0
    with tqdm(loader, desc=f"  {label} {ep+1:3d}/{total}", leave=False) as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * x.size(0)
            pred = model(x).argmax(1)
            tc  += pred.eq(y).sum().item()
            tn  += x.size(0)
            pbar.set_postfix(acc=f"{tc/tn:.3f}")
    return tl, tc, tn


def _log(ep, total, tr_loss, tr_acc, val_m, opt, history, phase):
    lr = opt.param_groups[0]["lr"]
    print(f"  E{ep:3d}/{total} | "
          f"tr={tr_acc:.4f} val={val_m['accuracy']:.4f} "
          f"loss={val_m['loss']:.4f} lr={lr:.5f}")
    history["epoch"].append(ep)
    history["phase"].append(phase)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_m["loss"])
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(val_m["accuracy"])
    history["lr"].append(lr)


def _maybe_save(model, path, ep, val_m, cfg,
                best_val_acc, best_epoch, no_improve):
    if val_m["accuracy"] > best_val_acc:
        best_val_acc = val_m["accuracy"]
        best_epoch   = ep
        no_improve   = 0
        torch.save({
            "state_dict": model.state_dict(),
            "metadata": {
                "epoch":    ep,
                "val_acc":  best_val_acc,
                "val_loss": val_m["loss"],
                "config":   cfg,
            },
        }, path)
    else:
        no_improve += 1
    return best_val_acc, best_epoch, no_improve


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Pruning smoke test
# ─────────────────────────────────────────────────────────────────────────────

def verify_pruning(model: ResNet18Prunable, device: torch.device) -> None:
    print("\n  [prune] Verifying 16-channel pruning interface …")
    dummy = torch.randn(2, 3, 32, 32).to(device)

    # forward_masked — full mask
    out = model.forward_masked(dummy, torch.ones(16).to(device))
    assert out.shape == (2, NUM_CLASSES), f"Bad shape: {out.shape}"

    # forward_masked — partial mask
    m = torch.ones(16); m[:8] = 0
    out2 = model.forward_masked(dummy, m.to(device))
    assert out2.shape == (2, NUM_CLASSES)

    # physical pruning with 9 retained channels (60% budget)
    pruned = build_pruned_model(model, list(range(9)), NUM_CLASSES).to(device)
    out3   = pruned(dummy)
    assert out3.shape == (2, NUM_CLASSES), f"Pruned output shape: {out3.shape}"

    # conv1 must stay at 16 channels
    assert model.conv1.weight.shape == (16, 3, 3, 3)

    print(f"  [prune] conv1 shape : {tuple(model.conv1.weight.shape)}  OK")
    print(f"  [prune] forward_masked (full mask)    OK — {tuple(out.shape)}")
    print(f"  [prune] forward_masked (half mask)    OK — {tuple(out2.shape)}")
    print(f"  [prune] build_pruned_model (9 ch)     OK — {tuple(out3.shape)}")
    print(f"  [prune] All checks PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(history: Dict, out_dir: str) -> None:
    epochs = history["epoch"]
    wu     = next((e for e, p in zip(epochs, history["phase"]) if p == 2), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss", lw=1.5)
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   lw=1.5)
    if wu: ax1.axvline(wu, color="orange", ls="--", alpha=0.5, label="Phase 2")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train acc", lw=1.5)
    ax2.plot(epochs, history["val_acc"],   label="Val acc",   lw=1.5)
    best_ep  = history["best_epoch"]
    best_acc = history["best_val_acc"]
    ax2.axvline(best_ep, color="red", ls="--", alpha=0.6,
                label=f"Best E{best_ep} ({best_acc:.3f})")
    if wu: ax2.axvline(wu, color="orange", ls="--", alpha=0.5, label="Phase 2")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
    plt.close(fig)

    # separate accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train acc", lw=1.5)
    ax.plot(epochs, history["val_acc"],   label="Val acc",   lw=1.5)
    ax.axvline(best_ep, color="red", ls="--", alpha=0.6,
               label=f"Best E{best_ep} ({best_acc:.3f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy (CIFAR-10, ResNet18 pretrained)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


def write_outputs(out_dir, metrics, history, cfg, subset_indices):
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
    keys = ["accuracy", "precision", "recall", "f1", "loss",
            "parameters", "memory_mb", "inference_ms", "training_time_s",
            "n_train", "n_val", "n_test"]
    pd.DataFrame([{k: mc.get(k) for k in keys}]).to_csv(
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

    # dataset_config.json
    ds_cfg = {
        "dataset":       "cifar10",
        "num_classes":   NUM_CLASSES,
        "class_names":   CIFAR10_CLASSES,
        "n_train":       len(subset_indices["train"]),
        "n_val":         len(subset_indices["val"]),
        "n_test":        len(subset_indices["test"]),
        "train_per_class": TRAIN_PER_CLS,
        "val_per_class":   VAL_PER_CLS,
        "test_per_class":  TEST_PER_CLS,
        "seed":            SEED,
        "train_source":    "official_cifar10_train",
        "val_source":      "official_cifar10_train_no_overlap",
        "test_source":     "official_cifar10_test",
        "normalization":   {
            "mean": list(CIFAR10_MEAN),
            "std":  list(CIFAR10_STD),
        },
        "augmentation_train": [
            "RandomHorizontalFlip",
            "RandomCrop(32,pad=4)",
            "RandAugment(n=2,m=7)",
        ],
        "augmentation_eval": [],
    }
    with open(os.path.join(out_dir, "dataset_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(ds_cfg, f, indent=2)

    # subset_indices.json
    with open(os.path.join(out_dir, "subset_indices.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "train_indices": subset_indices["train"],
            "val_indices":   subset_indices["val"],
            "test_indices":  subset_indices["test"],
        }, f, indent=2)

    # experiment_config.json
    with open(os.path.join(out_dir, "experiment_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # plots
    if history.get("epoch"):
        save_plots(history, out_dir)

    print(f"  [outputs] Written to {out_dir}/")


def write_diagnostic_report(out_dir, metrics, cfg, history):
    target = metrics["accuracy"] >= 0.70
    t_s    = metrics.get("training_time_s", 0)

    lines = [
        "# Diagnostic Report — Fast High-Accuracy Baseline",
        f"",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        "## Dataset",
        f"- **CIFAR-10** (10 classes): airplane, automobile, bird, cat, deer,",
        f"  dog, frog, horse, ship, truck.",
        f"- Why CIFAR-10: CIFAR-100 with 10k images = 100 samples/class —",
        f"  two prior runs capped at ~40% accuracy on CPU.",
        f"  CIFAR-10 with 3k images (300/class) is tractable for transfer learning.",
        f"- Train: {metrics['n_train']:,} | Val: {metrics['n_val']:,} | Test: {metrics['n_test']:,}",
        f"- Stratified, fixed seed={SEED}, no train/val/test overlap.",
        f"",
        "## Architecture: ResNet18Prunable",
        "| Layer | Details |",
        "|---|---|",
        "| **conv1** (PRUNABLE) | Conv2d(3, **16**, 3, pad=1) |",
        "| bn1 | BatchNorm2d(16) |",
        "| proj | Conv2d(16, 64, 1) — restores ResNet width |",
        "| bn_proj | BatchNorm2d(64) |",
        "| layer1-4 | Pretrained ResNet18 residual blocks |",
        "| avgpool | AdaptiveAvgPool2d(1,1) |",
        "| fc | Linear(512, 10) |",
        f"| **Total params** | {metrics['parameters']:,} |",
        f"",
        "## Training Configuration",
        f"| Phase 1 (frozen backbone) | {cfg.get('warmup_epochs',5)} epochs, LR={cfg.get('lr_phase1',0.01)}, SGD+Nesterov |",
        f"|---|---|",
        f"| Phase 2 (full fine-tune) | {cfg.get('epochs',25)-cfg.get('warmup_epochs',5)} epochs max, LR={cfg.get('lr_phase2',0.05)} |",
        f"| Scheduler | CosineAnnealingLR per phase |",
        f"| Early stopping | patience={cfg.get('patience',7)} epochs |",
        f"| Label smoothing | {cfg.get('label_smoothing',0.1)} |",
        f"| Augmentation | RandAugment(n=2,m=7) + RandomCrop + RandomFlip |",
        f"| Pretrained | ImageNet ResNet18 (torchvision) |",
        f"| Best epoch | {history.get('best_epoch','-')} |",
        f"| Training time | {t_s:.0f}s ({t_s/60:.1f} min) |",
        f"",
        "## Test Results (fixed 600-image test subset — evaluated ONCE)",
        f"| Test Accuracy | **{metrics['accuracy']:.4f}** |",
        f"|---|---|",
        f"| Macro Precision | {metrics['precision']:.4f} |",
        f"| Macro Recall | {metrics['recall']:.4f} |",
        f"| Macro F1 | {metrics['f1']:.4f} |",
        f"| Test Loss | {metrics['loss']:.4f} |",
        f"| Parameters | {metrics['parameters']:,} |",
        f"| Memory (MB) | {metrics['memory_mb']:.3f} |",
        f"| Inference (ms/sample) | {metrics['inference_ms']:.4f} |",
        f"",
        f"## 70-80% Target: **{'REACHED' if target else 'NOT REACHED'}**",
        f"Achieved: **{metrics['accuracy']:.2%}**",
        f"",
        "## Pruning Compatibility",
        "- `conv1` has exactly 16 output channels (verified).",
        "- `forward_masked(x, mask)` applies mask between conv1 and proj layer.",
        "- `build_pruned_model(model, retained_channels)` physically removes",
        "  channels from conv1/bn1 and the corresponding input slice of proj.",
        "- All downstream ResNet layers (layer1-4) are copied unchanged.",
        "- Compatible with classical pruning, exact QUBO, and QAOA.",
        f"",
        "## Transfer Learning Note",
        "Pretrained weights: ResNet18 trained on ImageNet (torchvision",
        "ResNet18_Weights.IMAGENET1K_V1). Only layer1-layer4 + avgpool use",
        "these weights. conv1, bn1, proj, bn_proj, and fc are randomly",
        "initialised and trained from scratch.",
        f"",
    ]

    if not target:
        lines += [
            "## Why target may not be reached",
            "- 600-image test set can have high variance.",
            "- CPU training limits epochs vs GPU runs.",
            "- RandAugment on small sets can occasionally hurt.",
        ]

    with open(os.path.join(out_dir, "diagnostic_report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  [report] diagnostic_report.md written")


def write_comparison_report(out_dir, metrics):
    prev_paths = {
        "Original SmallCNN (CIFAR-100, 50ep)":
            "results/baseline/metrics.json",
        "ImprovedSmallCNN (CIFAR-100, 40ep, 2k test)":
            "results/baseline_small/metrics.json",
    }
    prevs = {}
    for name, path in prev_paths.items():
        if os.path.exists(path):
            with open(path) as f:
                prevs[name] = json.load(f)

    metric_defs = [
        ("Test Accuracy",    "accuracy",      ".4f"),
        ("Macro F1",         "f1",            ".4f"),
        ("Test Loss",        "loss",          ".4f"),
        ("Parameters",       "parameters",    ",.0f"),
        ("Training Time(s)", "training_time_s", ".0f"),
    ]

    rows = []
    for label, key, fmt in metric_defs:
        row = {"Metric": label}
        for bname, b in prevs.items():
            v = b.get(key)
            row[bname] = f"{v:{fmt}}" if v is not None else "N/A"
        nv = metrics.get(key)
        row["ResNet18 CIFAR-10 (this run)"] = f"{nv:{fmt}}" if nv is not None else "N/A"
        rows.append(row)

    # Markdown
    if rows:
        cols = list(rows[0].keys())
        hdr  = "| " + " | ".join(cols) + " |"
        sep  = "|" + "|".join(["---" if i == 0 else "---:" for i in range(len(cols))]) + "|"
        lines = [
            "# Comparison Report\n",
            "> NOTE: Different datasets and test-set sizes — accuracy not directly comparable.",
            "",
            hdr, sep,
        ] + [
            "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
            for row in rows
        ]
        with open(os.path.join(out_dir, "comparison_report.md"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines))

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "comparison_report.csv"), index=False
    )
    print("  [report] comparison_report.md / .csv written")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG: Dict[str, Any] = {
    "data_dir":        "./data",
    "num_workers":     0,
    "seed":            SEED,
    "batch_size":      64,
    "epochs":          25,
    "warmup_epochs":   5,
    "lr_phase1":       0.01,
    "lr_phase2":       0.05,
    "momentum":        0.9,
    "weight_decay":    5e-4,
    "label_smoothing": 0.1,
    "patience":        7,
    "pretrained":      True,
    "dataset":         "cifar10",
    "num_classes":     NUM_CLASSES,
    "force_retrain":   False,
    "out_dir":         OUT_DIR,
    "ckpt_dir":        CKPT_DIR,
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=None)
    p.add_argument("--batch-size",    type=int,   default=None, dest="batch_size")
    p.add_argument("--lr2",           type=float, default=None)
    p.add_argument("--patience",      type=int,   default=None)
    p.add_argument("--force-retrain", action="store_true", dest="force_retrain")
    p.add_argument("--no-pretrained", action="store_true", dest="no_pretrained")
    args = p.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.epochs     is not None: cfg["epochs"]       = args.epochs
    if args.batch_size is not None: cfg["batch_size"]   = args.batch_size
    if args.lr2        is not None: cfg["lr_phase2"]    = args.lr2
    if args.patience   is not None: cfg["patience"]     = args.patience
    if args.force_retrain:          cfg["force_retrain"]= True
    if args.no_pretrained:          cfg["pretrained"]   = False

    # Reproducibility
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*62)
    print("  RobustQAOA — Fast High-Accuracy Baseline (CIFAR-10)")
    print("="*62)
    print(f"  Device    : {device}")
    print(f"  Pretrained: {cfg['pretrained']}")
    print(f"  Epochs    : {cfg['epochs']} (warmup={cfg['warmup_epochs']})")
    print(f"  Patience  : {cfg['patience']}")
    print("="*62)

    out_dir   = cfg["out_dir"]
    ckpt_dir  = cfg["ckpt_dir"]
    save_path = os.path.join(ckpt_dir, "baseline_model.pt")
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Subsets ───────────────────────────────────────────────────────────
    idx_path = os.path.join(out_dir, "subset_indices.json")
    if os.path.exists(idx_path) and not cfg["force_retrain"]:
        print(f"  [subset] Loading existing indices from {idx_path}")
        with open(idx_path) as f:
            d = json.load(f)
        train_idx = d["train_indices"]
        val_idx   = d["val_indices"]
        test_idx  = d["test_indices"]
    else:
        train_idx, val_idx, test_idx = build_subsets(cfg["data_dir"])

    subset_indices = {"train": train_idx, "val": val_idx, "test": test_idx}

    # ── Loaders ───────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_loaders(
        cfg["data_dir"], train_idx, val_idx, test_idx,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"  [data] {len(train_loader.dataset)} train / "
          f"{len(val_loader.dataset)} val / "
          f"{len(test_loader.dataset)} test")

    # ── Model ────────────────────────────────────────────────────────────
    model = ResNet18Prunable(
        num_classes=NUM_CLASSES,
        pretrained=cfg["pretrained"],
    ).to(device)
    n_params = count_parameters(model)
    print(f"  [model] ResNet18Prunable — {n_params:,} params")

    # ── Train or load ─────────────────────────────────────────────────────
    if os.path.exists(save_path) and not cfg["force_retrain"]:
        print(f"  [ckpt] Loading existing checkpoint …")
        pl = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(pl["state_dict"])
        meta    = pl.get("metadata", {})
        history = {
            "best_val_acc": meta.get("val_acc", 0),
            "best_epoch":   meta.get("epoch", 0),
            "epoch": [], "phase": [],
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [], "training_time_s": 0.0,
        }
    else:
        t0      = time.time()
        history = train(model, train_loader, val_loader, cfg, device, save_path)
        history["training_time_s"] = time.time() - t0
        # reload best
        pl = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(pl["state_dict"])
        print("  [ckpt] Best checkpoint reloaded.")

    # ── Shape check ───────────────────────────────────────────────────────
    # verify_model_shapes from model_utils checks for 100-class output.
    # Our model outputs (B, 10), so we do a direct check instead.
    model.eval()
    _dummy = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        _out = model(_dummy)
    assert _out.shape == (2, NUM_CLASSES), \
        f"Shape check FAILED: expected (2,{NUM_CLASSES}), got {tuple(_out.shape)}"
    print(f"  [shape] Output shape (2,{NUM_CLASSES}) OK")

    # ── Pruning check ─────────────────────────────────────────────────────
    verify_pruning(model, device)

    # ── Test evaluation (runs ONCE) ───────────────────────────────────────
    print("\n  [eval] Evaluating on fixed test subset (600 images) …")
    test_m = eval_loader(model, test_loader, device)
    test_m.update({
        "parameters":       n_params,
        "memory_mb":        model_memory_mb(model),
        "inference_ms":     measure_inference_time(model, device),
        "training_time_s":  history.get("training_time_s", 0.0),
        "best_val_acc":     history.get("best_val_acc", 0.0),
        "best_epoch":       history.get("best_epoch", 0),
        "device":           str(device),
        "seed":             cfg["seed"],
        "n_train":          len(train_idx),
        "n_val":            len(val_idx),
        "n_test":           len(test_idx),
        "dataset":          "cifar10",
        "architecture":     "ResNet18Prunable",
    })

    # ── Save all outputs ──────────────────────────────────────────────────
    write_outputs(out_dir, test_m, history, cfg, subset_indices)
    write_diagnostic_report(out_dir, test_m, cfg, history)
    write_comparison_report(out_dir, test_m)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FINAL RESULTS — CIFAR-10, ResNet18 pretrained")
    print(f"  {'─'*50}")
    print(f"  Test Accuracy  : {test_m['accuracy']:.4f}  ({test_m['accuracy']:.1%})")
    print(f"  Macro Precision: {test_m['precision']:.4f}")
    print(f"  Macro Recall   : {test_m['recall']:.4f}")
    print(f"  Macro F1       : {test_m['f1']:.4f}")
    print(f"  Test Loss      : {test_m['loss']:.4f}")
    print(f"  Parameters     : {test_m['parameters']:,}")
    print(f"  Memory         : {test_m['memory_mb']:.2f} MB")
    print(f"  Inference      : {test_m['inference_ms']:.3f} ms/sample")
    t_s = test_m["training_time_s"]
    print(f"  Training Time  : {t_s:.0f}s  ({t_s/60:.1f} min)")
    print(f"  70% target     : {'REACHED' if test_m['accuracy']>=0.70 else 'NOT REACHED'}")
    print(f"{'='*62}")

    print("\n  Files written:")
    for fname in [
        "metrics.json", "metrics.csv", "training_history.csv",
        "training_curve.png", "accuracy_curve.png",
        "dataset_config.json", "subset_indices.json",
        "experiment_config.json", "diagnostic_report.md",
        "comparison_report.md", "comparison_report.csv",
        "checkpoints/baseline_model.pt",
    ]:
        fp    = os.path.join(out_dir, fname)
        mark  = "OK" if os.path.exists(fp) else "MISSING"
        print(f"    [{mark}] {fname}")


if __name__ == "__main__":
    main()
