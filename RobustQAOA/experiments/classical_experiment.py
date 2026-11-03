"""
Classical / conventional pruning experiment.

Uses greedy importance-based channel selection (see pruning/classical_pruning.py).
Physically constructs the pruned model and evaluates all required metrics.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple

from pruning.classical_pruning import run_classical_pruning
from pruning.pruning_utils import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
)
from models.model_utils import count_parameters, model_memory_mb, measure_inference_time
from training.evaluate import evaluate_full
from robustness.robustness_score import evaluate_model_robustness


def run_classical_experiment(
    baseline_model: torch.nn.Module,
    importance:     np.ndarray,
    robustness_scores: np.ndarray,
    cfg:            Dict[str, Any],
    test_loader,
    device:         torch.device,
    baseline_metrics: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """
    Full classical pruning pipeline.

    Args:
        baseline_model    : trained SmallCNN (not modified)
        importance        : c_i array, shape (16,)
        robustness_scores : r_i array, shape (16,)
        cfg               : config dict
        test_loader       : CIFAR-100 test DataLoader
        device            : torch device
        baseline_metrics  : optional dict for computing reductions

    Returns:
        results      : dict of all metrics
        pruned_model : physically constructed pruned model
    """
    print(f"\n{'='*60}")
    print("  CLASSICAL PRUNING EXPERIMENT")
    print(f"{'='*60}")

    num_channels    = cfg.get("num_prunable_channels", 16)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)

    # ── Run classical pruning ─────────────────────────────────────────────────
    results, pruned_model = run_classical_pruning(
        baseline_model, importance, robustness_scores,
        cfg, test_loader, device,
    )

    # ── Evaluate pruned model ─────────────────────────────────────────────────
    clean_metrics = evaluate_full(pruned_model, test_loader, device)
    rob_metrics   = evaluate_model_robustness(pruned_model, test_loader, cfg, device)

    # ── Memory budget verification ────────────────────────────────────────────
    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)
    mask = results["mask"]
    feasible = check_feasibility(mask, w, M)
    if not feasible:
        print(f"  [classical] WARNING: Memory constraint VIOLATED. "
              f"Used: {float(np.dot(w, mask)):.0f} > Budget: {M:.0f}")

    # ── Compute reductions ────────────────────────────────────────────────────
    n_params = clean_metrics["parameters"]
    mem_mb   = clean_metrics["memory_mb"]

    param_reduction = 0.0
    mem_reduction   = 0.0
    if baseline_metrics:
        base_params = baseline_metrics.get("parameters", 1)
        base_mem    = baseline_metrics.get("memory_mb",  1.0)
        param_reduction = (1.0 - n_params / base_params) * 100.0
        mem_reduction   = (1.0 - mem_mb   / base_mem)    * 100.0

    # ── Compile full results ──────────────────────────────────────────────────
    results.update({
        **clean_metrics,
        **rob_metrics,
        "param_reduction_pct": param_reduction,
        "mem_reduction_pct":   mem_reduction,
        "memory_budget_bytes": M,
        "memory_used_bytes":   float(np.dot(w, mask)),
    })

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("results/classical", exist_ok=True)

    def _serializable(v):
        if isinstance(v, np.ndarray):      return v.tolist()
        if isinstance(v, (np.floating, np.integer)): return float(v)
        return v

    metrics_out = {k: _serializable(v) for k, v in results.items()}
    with open("results/classical/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open("results/classical/selected_channels.json", "w") as f:
        json.dump({
            "retained": results["retained_channels"],
            "pruned":   results["pruned_channels"],
            "mask":     mask.tolist(),
        }, f, indent=2)

    torch.save(
        {"state_dict": pruned_model.state_dict(), "retained": results["retained_channels"]},
        "results/classical/model.pt",
    )

    _print_results(results)
    return results, pruned_model


def _print_results(r: Dict[str, Any]) -> None:
    print(f"\n  CLASSICAL PRUNING RESULTS")
    print(f"  {'─'*40}")
    print(f"  Retained channels  : {r.get('retained_channels')}")
    print(f"  Pruned channels    : {r.get('pruned_channels')}")
    print(f"  Clean Accuracy     : {r.get('accuracy', 'N/A'):.4f}")
    print(f"  Macro F1           : {r.get('f1', 'N/A'):.4f}")
    print(f"  Parameters         : {r.get('parameters', 'N/A'):,}")
    print(f"  Param Reduction    : {r.get('param_reduction_pct', 0):.2f}%")
    print(f"  Memory (MB)        : {r.get('memory_mb', 'N/A'):.4f}")
    print(f"  FGSM Accuracy      : {r.get('fgsm_accuracy', 'N/A'):.4f}")
    print(f"  PGD Accuracy       : {r.get('pgd_accuracy', 'N/A'):.4f}")
    print(f"  Robust Accuracy    : {r.get('robust_accuracy', 'N/A'):.4f}")
    print(f"  Runtime (s)        : {r.get('runtime_s', 'N/A'):.4f}")
    print(f"  QUBO Objective     : {r.get('qubo_objective', 'N/A'):.4f}")
