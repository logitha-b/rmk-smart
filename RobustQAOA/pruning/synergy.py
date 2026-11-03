"""
Pairwise channel synergy matrix computation.

For every pair (i, j) with i < j evaluate:
    f(full)   = accuracy of full model
    f(-i)     = accuracy with channel i zeroed
    f(-j)     = accuracy with channel j zeroed
    f(-i,-j)  = accuracy with channels i and j zeroed

Synergy formula (documented):
    effect_i    = f(full) - f(-i)
    effect_j    = f(full) - f(-j)
    effect_ij   = f(full) - f(-i,-j)   [combined effect of removing both]
    synergy_ij  = effect_ij - effect_i - effect_j

    Interpretation:
      synergy_ij > 0 → removing both is WORSE than the sum of individuals
                        (the channels complement each other; keeping both is
                        more than the sum of parts)
      synergy_ij < 0 → removing both is BETTER than sum (channels overlap;
                        they provide redundant information)
      synergy_ij ≈ 0 → channels act independently

The returned matrix Q is symmetric with Q[i,i] = 0.

Computational cost: O(n^2 / 2) ablation evaluations.
For n=16 this requires 120 pairwise evaluations plus 16 single-channel
evaluations (which may already be available from importance.py).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple

from pruning.importance import _collect_subset, _accuracy_masked


def compute_synergy_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_channels: int = 16,
    n_samples: int = 500,
    seed: int = 42,
    baseline_acc: float = None,
    single_ablation_accs: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the 16×16 pairwise synergy matrix.

    Args:
        model                : trained SmallCNN
        loader               : validation DataLoader
        device               : torch device
        num_channels         : 16
        n_samples            : validation subset size
        seed                 : random seed
        baseline_acc         : f(full) — reuse from importance if available
        single_ablation_accs : array of f(-i) values — reuse if available

    Returns:
        Q : np.ndarray of shape (num_channels, num_channels)
            Symmetric, zeros on diagonal.
    """
    model.eval()

    # ── Reuse cached subset ───────────────────────────────────────────────────
    all_inputs, all_targets = _collect_subset(loader, n_samples, seed, device)

    # ── Full model accuracy ───────────────────────────────────────────────────
    if baseline_acc is None:
        baseline_acc = _accuracy_masked(model, all_inputs, all_targets, mask=None)
        print(f"  [synergy] Full-model accuracy: {baseline_acc:.4f}")

    # ── Single-channel ablation accuracies ───────────────────────────────────
    if single_ablation_accs is None:
        single_ablation_accs = np.zeros(num_channels)
        for i in range(num_channels):
            mask                     = torch.ones(num_channels)
            mask[i]                  = 0.0
            single_ablation_accs[i]  = _accuracy_masked(
                model, all_inputs, all_targets, mask
            )

    # effect_i = f(full) - f(-i)
    effect_single = baseline_acc - single_ablation_accs   # shape (n,)

    # ── Pairwise ablation ────────────────────────────────────────────────────
    Q = np.zeros((num_channels, num_channels), dtype=np.float64)

    pairs = [(i, j) for i in range(num_channels) for j in range(i + 1, num_channels)]

    for (i, j) in tqdm(pairs, desc="  Pairwise synergy"):
        mask    = torch.ones(num_channels)
        mask[i] = 0.0
        mask[j] = 0.0
        acc_ij  = _accuracy_masked(model, all_inputs, all_targets, mask)

        effect_ij    = baseline_acc - acc_ij
        synergy_val  = effect_ij - effect_single[i] - effect_single[j]

        Q[i, j] = synergy_val
        Q[j, i] = synergy_val   # symmetric

    print(f"  [synergy] Q matrix range: [{Q.min():.4f}, {Q.max():.4f}]")
    return Q
