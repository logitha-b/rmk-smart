"""
Channel importance calculation via ablation.

For each of the 16 channels i in conv1:
    c_i = f(full) - f(-i)

where f(·) is the top-1 accuracy on the validation subset.

The full-model accuracy is evaluated once and reused for all ablations.
All ablation evaluations use exactly the same validation subset to ensure
comparability.

Formula (documented):
    effect_i = accuracy(full model) - accuracy(model with channel i zeroed)
    c_i = max(0, effect_i)   [negative values set to 0; channel adds nothing]

Note: c_i may be negative in theory if ablating a channel improves accuracy
(channel is actively harmful). We retain the raw signed value in the returned
array for transparency and only clip during QUBO construction if desired.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple


def compute_channel_importance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_channels: int = 16,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Compute channel importance scores via single-channel ablation.

    Args:
        model        : trained SmallCNN (in eval mode, not modified in-place)
        loader       : validation DataLoader
        device       : torch device
        num_channels : number of prunable channels (16)
        n_samples    : size of the fixed validation subset
        seed         : random seed for subset selection

    Returns:
        importance   : np.ndarray of shape (num_channels,), values = c_i
        baseline_acc : float, accuracy of the full model on the subset
    """
    model.eval()

    # ── Build fixed validation subset ────────────────────────────────────────
    all_inputs, all_targets = _collect_subset(loader, n_samples, seed, device)

    # ── Full model accuracy ───────────────────────────────────────────────────
    baseline_acc = _accuracy_masked(model, all_inputs, all_targets, mask=None)
    print(f"  [importance] Full-model accuracy on subset: {baseline_acc:.4f}")

    # ── Per-channel ablation ─────────────────────────────────────────────────
    importance = np.zeros(num_channels, dtype=np.float64)
    for i in tqdm(range(num_channels), desc="  Channel ablation"):
        mask = torch.ones(num_channels, dtype=torch.float32)
        mask[i] = 0.0
        acc_minus_i  = _accuracy_masked(model, all_inputs, all_targets, mask)
        importance[i] = baseline_acc - acc_minus_i   # c_i = f(full) - f(-i)

    print(f"  [importance] c_i values: {np.round(importance, 4)}")
    print(f"  [importance] Min: {importance.min():.4f}  Max: {importance.max():.4f}")

    return importance, baseline_acc


def _collect_subset(
    loader: DataLoader,
    n_samples: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect a fixed random subset from a DataLoader onto device."""
    all_inputs, all_targets = [], []
    for inputs, targets in loader:
        all_inputs.append(inputs)
        all_targets.append(targets)

    all_inputs_t  = torch.cat(all_inputs,  dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)

    rng = np.random.default_rng(seed)
    n   = min(n_samples, all_inputs_t.size(0))
    idx = rng.choice(all_inputs_t.size(0), size=n, replace=False)

    return all_inputs_t[idx].to(device), all_targets_t[idx].to(device)


def _accuracy_masked(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask,                     # None = full model; else torch.Tensor of shape (16,)
    batch_size: int = 128,
) -> float:
    """Evaluate accuracy with an optional channel mask applied to conv1."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            end  = min(start + batch_size, inputs.size(0))
            inp  = inputs[start:end]
            tgt  = targets[start:end]

            if mask is None:
                out = model(inp)
            else:
                out = model.forward_masked(inp, mask)

            pred    = out.argmax(dim=1)
            correct += pred.eq(tgt).sum().item()
            total   += inp.size(0)

    return correct / total if total > 0 else 0.0
