"""
Training routines for the SmallCNN on CIFAR-100.

Supports full and debug mode (controlled by config).
Records training/validation loss and accuracy per epoch.
Saves the best checkpoint by validation accuracy.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar100_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Download CIFAR-100 and return (train_loader, test_loader).

    In debug mode a small random subset is used for speed.
    """
    data_dir   = cfg.get("data_dir", "./data")
    batch_size = cfg.get("batch_size", 128)
    num_workers = cfg.get("num_workers", 2)
    seed       = cfg.get("seed", 42)
    debug      = cfg.get("debug_mode", False)

    # CIFAR-100 normalisation statistics
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if debug:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(len(train_dataset), size=2000, replace=False).tolist()
        test_idx  = rng.choice(len(test_dataset),  size=500,  replace=False).tolist()
        train_dataset = Subset(train_dataset, train_idx)
        test_dataset  = Subset(test_dataset,  test_idx)
        print(f"  [debug] Using {len(train_dataset)} train / {len(test_dataset)} test samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    with tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            predicted   = outputs.argmax(dim=1)
            correct    += predicted.eq(targets).sum().item()
            total      += inputs.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    avg_loss = total_loss / total
    accuracy = correct    / total
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────────────────
# Full training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    cfg: Dict[str, Any],
    device: torch.device,
    save_path: str,
) -> Dict[str, Any]:
    """
    Full training loop for SmallCNN on CIFAR-100.

    Args:
        model     : untrained SmallCNN
        cfg       : configuration dict
        device    : torch device
        save_path : path to save best checkpoint

    Returns:
        history dict with lists: train_loss, val_loss, train_acc, val_acc
    """
    from training.evaluate import evaluate_model  # local import to avoid circularity

    epochs   = cfg.get("epochs", 50)
    lr       = cfg.get("learning_rate", 0.05)
    momentum = cfg.get("momentum", 0.9)
    wd       = cfg.get("weight_decay", 0.0005)
    sched_t  = cfg.get("scheduler", "cosine")

    # Build loaders
    train_loader, val_loader = get_cifar100_loaders(cfg)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # Scheduler
    if sched_t == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_t == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    else:
        scheduler = None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    [],
    }

    best_val_acc  = -1.0
    best_epoch    = -1
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n{'='*60}")
    print("  TRAINING BASELINE MODEL")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Device: {device}")
    print(f"{'='*60}")

    t_start = time.time()

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc  = val_metrics["accuracy"]

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if scheduler:
            scheduler.step()

        print(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "epoch":       best_epoch,
                        "val_acc":     best_val_acc,
                        "val_loss":    val_loss,
                        "config":      cfg,
                    },
                },
                save_path,
            )

    elapsed = time.time() - t_start
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"  Checkpoint: {save_path}")

    history["best_val_acc"] = best_val_acc
    history["best_epoch"]   = best_epoch
    history["training_time_s"] = elapsed

    return history
