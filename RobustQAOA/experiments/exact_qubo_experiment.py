"""
Exact QUBO experiment.

Enumerates all 2^16 bitstrings and selects the optimal feasible solution.
This serves as the ground-truth optimizer for the QUBO formulation.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple

from optimization.qubo       import build_qubo_matrix, evaluate_qubo_vectorized
from optimization.exact_solver import exact_qubo_solve
from pruning.pruning_utils   import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
)
from models.pruned_model     import build_pruned_model
from models.model_utils      import count_parameters, model_memory_mb, measure_inference_time
from training.evaluate       import evaluate_full
from robustness.robustness_score import evaluate_model_robustness


def run_exact_qubo_experiment(
    baseline_model:    torch.nn.Module,
    importance:        np.ndarray,
    robustness_scores: np.ndarray,
    synergy:           np.ndarray,
    cfg:               Dict[str, Any],
    test_loader,
    device:            torch.device,
    baseline_metrics:  Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """
    Run the exact QUBO experiment.

    Returns:
        results      : dict of all metrics
        pruned_model : physically constructed pruned model
    """
    print(f"\n{'='*60}")
    print("  EXACT QUBO EXPERIMENT")
    print(f"{'='*60}")

    num_channels    = cfg.get("num_prunable_channels", 16)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)
    alpha           = cfg.get("alpha",          1.0)
    beta            = cfg.get("beta",            1.0)
    gamma           = cfg.get("gamma",           0.5)
    lam             = cfg.get("lambda_penalty",  10.0)

    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)

    # ── Build QUBO ────────────────────────────────────────────────────────────
    lin, quad = build_qubo_matrix(importance, robustness_scores, synergy, w, M,
                                   alpha, beta, gamma, lam)

    # ── Exact solve ───────────────────────────────────────────────────────────
    solver_result = exact_qubo_solve(lin, quad, w, M, num_channels)

    retained = solver_result["retained_channels"]
    mask     = solver_result["best_mask"]

    # ── Build pruned model ────────────────────────────────────────────────────
    pruned_model = build_pruned_model(baseline_model, retained, num_classes=100)
    pruned_model = pruned_model.to(device)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    clean_metrics = evaluate_full(pruned_model, test_loader, device)
    rob_metrics   = evaluate_model_robustness(pruned_model, test_loader, cfg, device)

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

    results = {
        "method":             "exact_qubo",
        "retained_channels":  retained,
        "pruned_channels":    solver_result["pruned_channels"],
        "mask":               mask,
        "runtime_s":          solver_result["runtime_s"],
        "feasible":           solver_result["feasible"],
        "qubo_objective":     solver_result["best_objective"],
        "n_evaluated":        solver_result["n_evaluated"],
        "n_feasible":         solver_result["n_feasible"],
        **clean_metrics,
        **rob_metrics,
        "param_reduction_pct": param_reduction,
        "mem_reduction_pct":   mem_reduction,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("results/exact_qubo", exist_ok=True)

    def _ser(v):
        if isinstance(v, np.ndarray):              return v.tolist()
        if isinstance(v, (np.floating, np.integer)): return float(v)
        return v

    metrics_out = {k: _ser(v) for k, v in results.items()}
    with open("results/exact_qubo/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open("results/exact_qubo/selected_channels.json", "w") as f:
        json.dump({"retained": retained, "pruned": results["pruned_channels"],
                   "mask": mask.tolist()}, f, indent=2)

    with open("results/exact_qubo/solver_results.json", "w") as f:
        json.dump({k: _ser(v) for k, v in solver_result.items()}, f, indent=2)

    _print_results(results)
    return results, pruned_model


def _print_results(r: Dict[str, Any]) -> None:
    print(f"\n  EXACT QUBO RESULTS")
    print(f"  {'─'*40}")
    print(f"  Retained channels  : {r.get('retained_channels')}")
    print(f"  Clean Accuracy     : {r.get('accuracy', 'N/A'):.4f}")
    print(f"  Macro F1           : {r.get('f1', 'N/A'):.4f}")
    print(f"  QUBO Objective     : {r.get('qubo_objective', 'N/A'):.6f}")
    print(f"  Runtime (s)        : {r.get('runtime_s', 'N/A'):.4f}")
    print(f"  Feasible           : {r.get('feasible')}")
