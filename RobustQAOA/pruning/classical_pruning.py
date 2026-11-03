"""
Classical / conventional greedy channel pruning.

This implements a standard importance-based greedy selection procedure
that is a genuine alternative to QUBO/QAOA optimization.

It does NOT call the exact QUBO solver or use the QUBO objective.

Algorithm:
    1. Compute a combined importance score for each channel:
           score_i = alpha * c_i + beta * r_i
       where c_i = clean-performance contribution,
             r_i = adversarial robustness contribution.
    2. Sort channels by score_i in descending order.
    3. Greedily retain channels (in order of decreasing score) until
       the next channel would violate the memory budget.
    4. Construct the physically pruned model by calling build_pruned_model.
    5. Evaluate the pruned model.

This is the conventional pruning output used as the classical baseline.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Any

from pruning.pruning_utils import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
    mask_to_channel_list,
    channel_list_to_mask,
)
from models.pruned_model import build_pruned_model


def classical_greedy_selection(
    importance: np.ndarray,
    robustness: np.ndarray,
    alpha: float,
    beta: float,
    num_channels: int,
    budget_fraction: float,
) -> Tuple[np.ndarray, float, List[int]]:
    """
    Greedy importance-based channel selection.

    Args:
        importance      : c_i array, shape (num_channels,)
        robustness      : r_i array, shape (num_channels,)
        alpha           : clean-performance weight
        beta            : robustness weight
        num_channels    : 16
        budget_fraction : memory budget fraction

    Returns:
        mask       : binary np.ndarray of shape (num_channels,)
        runtime_s  : wall-clock seconds for optimization
        retained   : list of retained channel indices
    """
    t_start = time.time()

    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)

    # Combined score for each channel
    scores = alpha * importance + beta * robustness

    # Sort descending
    order  = np.argsort(scores)[::-1]

    mask   = np.zeros(num_channels, dtype=np.float64)
    used   = 0.0

    for idx in order:
        cost = w[idx]
        if used + cost <= M:
            mask[idx] = 1.0
            used      += cost

    runtime_s = time.time() - t_start
    retained  = mask_to_channel_list(mask)

    print(f"  [classical] Retained channels : {retained}")
    print(f"  [classical] Memory used       : {used:.0f} / {M:.0f} bytes")
    print(f"  [classical] Runtime           : {runtime_s:.4f}s")

    return mask, runtime_s, retained


def run_classical_pruning(
    baseline_model: nn.Module,
    importance: np.ndarray,
    robustness: np.ndarray,
    cfg: Dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full classical pruning pipeline:
        1. Greedy selection
        2. Build pruned model
        3. Evaluate

    Returns a results dict.
    """
    from training.evaluate import evaluate_full
    from models.model_utils import count_parameters, model_memory_mb, measure_inference_time

    alpha           = cfg.get("alpha",                  1.0)
    beta            = cfg.get("beta",                   1.0)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)
    num_channels    = cfg.get("num_prunable_channels",  16)
    seed            = cfg.get("seed",                   42)

    # ── Greedy channel selection ─────────────────────────────────────────────
    mask, runtime_s, retained = classical_greedy_selection(
        importance, robustness, alpha, beta, num_channels, budget_fraction
    )

    # ── Build physically pruned model ────────────────────────────────────────
    pruned_model = build_pruned_model(baseline_model, retained, num_classes=100)
    pruned_model = pruned_model.to(device)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate_full(pruned_model, loader, device)

    # ── Compute QUBO objective for the selected mask (for comparison table) ──
    # This is informational — the classical method did NOT optimise this.
    qubo_obj = _compute_dummy_qubo_obj(mask, importance, robustness, cfg)

    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)

    results = {
        "method":           "classical",
        "retained_channels": retained,
        "pruned_channels":   [i for i in range(num_channels) if i not in retained],
        "mask":             mask,
        "runtime_s":        runtime_s,
        "feasible":         check_feasibility(mask, w, M),
        "qubo_objective":   qubo_obj,
        **metrics,
    }
    return results, pruned_model


def _compute_dummy_qubo_obj(
    mask: np.ndarray,
    importance: np.ndarray,
    robustness: np.ndarray,
    cfg: Dict[str, Any],
) -> float:
    """
    Evaluate the linear part of the QUBO objective for a given mask.
    This is used for comparison purposes only.
    The pairwise term is omitted here since synergy is not used in classical selection.
    """
    alpha  = cfg.get("alpha",  1.0)
    beta   = cfg.get("beta",   1.0)
    return float(alpha * np.dot(importance, mask) + beta * np.dot(robustness, mask))
