"""
Attack pool construction.

The attack pool is a fixed set of adversarial examples generated against
the current model.  It is used to:
  1. Measure per-channel robustness (r_i)
  2. Feed back into the next defender-attacker iteration

Both FGSM and PGD attacks are included.

The pool is stored as a list of (adversarial_input, true_label) tensors.
The pool is reproducible: the same clean samples and random seeds produce
the same pool for a given model.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

from attacks.fgsm import fgsm_attack
from attacks.pgd  import pgd_attack


AttackPool = List[Tuple[torch.Tensor, torch.Tensor]]


def build_attack_pool(
    model: nn.Module,
    loader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> AttackPool:
    """
    Generate adversarial examples and return the attack pool.

    The pool contains n_pool = min(attack_pool_size, dataset_size) examples,
    split 50/50 between FGSM and PGD.

    Args:
        model  : current model
        loader : DataLoader (clean validation set)
        cfg    : configuration dict
        device : torch device

    Returns:
        pool : list of (adv_tensor, label_tensor) — each item is a single sample
    """
    fgsm_eps    = cfg.get("fgsm_epsilon",    0.03)
    pgd_eps     = cfg.get("pgd_epsilon",     0.03)
    pgd_step    = cfg.get("pgd_step_size",   0.007)
    pgd_steps   = cfg.get("pgd_steps",       10)
    pool_size   = cfg.get("attack_pool_size", 200)
    seed        = cfg.get("seed",            42)

    half = pool_size // 2

    model.eval()
    pool: AttackPool = []

    # ── Collect clean samples ────────────────────────────────────────────────
    all_inputs, all_targets = [], []
    for inputs, targets in loader:
        all_inputs.append(inputs)
        all_targets.append(targets)
    all_inputs_t  = torch.cat(all_inputs,  dim=0)
    all_targets_t = torch.cat(all_targets, dim=0)

    rng = np.random.default_rng(seed)
    n   = min(pool_size, all_inputs_t.size(0))
    idx = rng.choice(all_inputs_t.size(0), size=n, replace=False)

    sel_inputs  = all_inputs_t[idx]
    sel_targets = all_targets_t[idx]

    n_fgsm = min(half, n)
    n_pgd  = n - n_fgsm

    # ── FGSM pool ────────────────────────────────────────────────────────────
    batch_size = 64
    for start in tqdm(range(0, n_fgsm, batch_size), desc="  Building FGSM pool"):
        end   = min(start + batch_size, n_fgsm)
        inp   = sel_inputs[start:end].to(device)
        tgt   = sel_targets[start:end].to(device)
        adv   = fgsm_attack(model, inp, tgt, fgsm_eps, device)
        for i in range(adv.size(0)):
            pool.append((adv[i:i+1].cpu(), tgt[i:i+1].cpu()))

    # ── PGD pool ─────────────────────────────────────────────────────────────
    for start in tqdm(range(n_fgsm, n_fgsm + n_pgd, batch_size), desc="  Building PGD pool"):
        end  = min(start + batch_size, n)
        inp  = sel_inputs[start:end].to(device)
        tgt  = sel_targets[start:end].to(device)
        adv  = pgd_attack(model, inp, tgt, pgd_eps, pgd_step, pgd_steps, device)
        for i in range(adv.size(0)):
            pool.append((adv[i:i+1].cpu(), tgt[i:i+1].cpu()))

    print(f"  [pool] Built {len(pool)} adversarial examples "
          f"({n_fgsm} FGSM + {n_pgd} PGD)")
    return pool


def evaluate_on_pool(
    model: nn.Module,
    pool: AttackPool,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Evaluate model accuracy on the pre-generated attack pool.

    Returns:
        Dict with 'accuracy' and 'avg_loss'.
    """
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    # Batch the pool
    for start in range(0, len(pool), batch_size):
        batch  = pool[start:start + batch_size]
        adv    = torch.cat([x for x, _ in batch], dim=0).to(device)
        labels = torch.cat([y for _, y in batch], dim=0).to(device)

        with torch.no_grad():
            out  = model(adv)
            loss = criterion(out, labels)
            pred = out.argmax(dim=1)
            correct    += pred.eq(labels).sum().item()
            total      += adv.size(0)
            total_loss += loss.item() * adv.size(0)

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "avg_loss": total_loss / total if total > 0 else 0.0,
    }
