"""
Per-channel adversarial robustness score computation.

Robustness score r_i for channel i:
    r_i = robustness_baseline - robustness_without_channel_i

where:
    robustness_baseline        = accuracy of full model on attack pool
    robustness_without_ch_i    = accuracy of model with channel i zeroed
                                  evaluated on the SAME attack pool

Formula (documented):
    r_i = Acc_robust(full) - Acc_robust(model with channel i zeroed)

Interpretation:
    r_i > 0  → channel i contributes positively to adversarial robustness
    r_i < 0  → removing channel i IMPROVES robustness (redundant / misleading)
    r_i ≈ 0  → channel i is robustness-neutral

The raw signed values are retained for use in the QUBO objective.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Dict, Any

from robustness.attack_pool import AttackPool, evaluate_on_pool


def compute_robustness_scores(
    model: nn.Module,
    pool: AttackPool,
    device: torch.device,
    num_channels: int = 16,
) -> Tuple[np.ndarray, float]:
    """
    Compute per-channel robustness scores from a pre-built attack pool.

    Args:
        model        : trained SmallCNN (not modified)
        pool         : pre-generated attack pool
        device       : torch device
        num_channels : number of prunable channels (16)

    Returns:
        robustness : np.ndarray of shape (num_channels,), values = r_i
        baseline_rob : float, robustness accuracy of full model on pool
    """
    model.eval()

    # ── Full model robustness on pool ────────────────────────────────────────
    baseline_result = evaluate_on_pool(model, pool, device)
    baseline_rob    = baseline_result["accuracy"]
    print(f"  [robustness] Full-model pool accuracy: {baseline_rob:.4f}")

    # ── Per-channel ablation on pool ─────────────────────────────────────────
    robustness = np.zeros(num_channels, dtype=np.float64)

    for i in tqdm(range(num_channels), desc="  Robustness ablation"):
        mask = torch.ones(num_channels, dtype=torch.float32)
        mask[i] = 0.0
        acc_minus_i = _pool_accuracy_masked(model, pool, device, mask)
        robustness[i] = baseline_rob - acc_minus_i

    print(f"  [robustness] r_i values: {np.round(robustness, 4)}")
    print(f"  [robustness] Min: {robustness.min():.4f}  Max: {robustness.max():.4f}")

    return robustness, baseline_rob


def _pool_accuracy_masked(
    model: nn.Module,
    pool: AttackPool,
    device: torch.device,
    mask: torch.Tensor,
    batch_size: int = 64,
) -> float:
    """
    Evaluate accuracy on the attack pool using a channel mask on conv1.

    Args:
        model     : SmallCNN with forward_masked method
        pool      : list of (adv_tensor, label_tensor)
        device    : torch device
        mask      : float tensor of shape (num_channels,)
        batch_size: batch size for evaluation

    Returns:
        accuracy (float)
    """
    model.eval()
    correct, total = 0, 0

    for start in range(0, len(pool), batch_size):
        batch  = pool[start:start + batch_size]
        adv    = torch.cat([x for x, _ in batch], dim=0).to(device)
        labels = torch.cat([y for _, y in batch], dim=0).to(device)

        with torch.no_grad():
            out  = model.forward_masked(adv, mask)
            pred = out.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total   += adv.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_model_robustness(
    model: nn.Module,
    loader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a (pruned) model's robustness using fresh attacks.

    Computes FGSM accuracy, PGD accuracy, robust accuracy (min of both),
    and worst-case attack loss.

    Args:
        model  : pruned or full model
        loader : evaluation DataLoader
        cfg    : configuration dict
        device : torch device

    Returns:
        Dict with fgsm_accuracy, pgd_accuracy, robust_accuracy, worst_attack_loss
    """
    from attacks.fgsm import fgsm_accuracy
    from attacks.pgd  import pgd_accuracy
    import torch.nn as nn

    fgsm_eps  = cfg.get("fgsm_epsilon",    0.03)
    pgd_eps   = cfg.get("pgd_epsilon",     0.03)
    pgd_step  = cfg.get("pgd_step_size",   0.007)
    pgd_steps = cfg.get("pgd_steps",       10)
    n_samples = cfg.get("eval_samples",    500)

    fgsm_acc = fgsm_accuracy(model, loader, fgsm_eps, device, n_samples)
    pgd_acc  = pgd_accuracy(model, loader, pgd_eps, pgd_step, pgd_steps, device, n_samples)

    robust_acc = min(fgsm_acc, pgd_acc)

    # Worst-case attack loss: evaluate PGD loss (higher = worse for model)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, total = 0.0, 0

    from attacks.pgd import pgd_attack
    for inputs, targets in loader:
        if total >= n_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        adv = pgd_attack(model, inputs, targets, pgd_eps, pgd_step, pgd_steps, device)
        with torch.no_grad():
            out  = model(adv)
            loss = criterion(out, targets)
            total_loss += loss.item() * inputs.size(0)
            total      += inputs.size(0)

    worst_loss = total_loss / total if total > 0 else float('nan')

    return {
        "fgsm_accuracy":    fgsm_acc,
        "pgd_accuracy":     pgd_acc,
        "robust_accuracy":  robust_acc,
        "worst_attack_loss": worst_loss,
    }
