"""
QAOA pruning experiment.

Uses the QAOA optimizer (Qiskit Aer) to solve the same QUBO formulation
as the exact solver.  Physically constructs and evaluates the pruned model.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple

from optimization.qubo  import build_qubo_matrix, evaluate_qubo_vectorized
from optimization.qaoa  import run_qaoa
from pruning.pruning_utils import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
)
from models.pruned_model   import build_pruned_model
from models.model_utils    import count_parameters, model_memory_mb, measure_inference_time
from training.evaluate     import evaluate_full
from robustness.robustness_score import evaluate_model_robustness


def run_qaoa_experiment(
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
    Full QAOA pruning pipeline.

    Returns:
        results      : dict of all metrics
        pruned_model : physically constructed pruned model
    """
    print(f"\n{'='*60}")
    print("  QAOA EXPERIMENT")
    print(f"{'='*60}")

    num_channels    = cfg.get("num_prunable_channels", 16)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)
    alpha           = cfg.get("alpha",         1.0)
    beta            = cfg.get("beta",           1.0)
    gamma           = cfg.get("gamma",          0.5)
    lam             = cfg.get("lambda_penalty", 10.0)

    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)

    # ── Build QUBO (same as exact solver) ────────────────────────────────────
    lin, quad = build_qubo_matrix(importance, robustness_scores, synergy, w, M,
                                   alpha, beta, gamma, lam)

    # ── Run QAOA ─────────────────────────────────────────────────────────────
    qaoa_result = run_qaoa(lin, quad, w, M, cfg)

    retained  = qaoa_result["retained_channels"]
    mask      = qaoa_result["best_mask"]

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
        "method":             qaoa_result.get("method", "qaoa"),
        "retained_channels":  retained,
        "pruned_channels":    qaoa_result["pruned_channels"],
        "mask":               mask,
        "runtime_s":          qaoa_result["runtime_s"],
        "feasible":           qaoa_result["feasible"],
        "qubo_objective":     qaoa_result["best_objective"],
        "qaoa_depth":         qaoa_result.get("qaoa_depth"),
        "qaoa_shots":         qaoa_result.get("qaoa_shots"),
        "qaoa_optimizer":     qaoa_result.get("qaoa_optimizer"),
        "qaoa_iterations":    qaoa_result.get("qaoa_iterations"),
        "circuit_info":       qaoa_result.get("circuit_info", {}),
        **clean_metrics,
        **rob_metrics,
        "param_reduction_pct": param_reduction,
        "mem_reduction_pct":   mem_reduction,
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("results/qaoa", exist_ok=True)

    def _ser(v):
        if isinstance(v, np.ndarray):              return v.tolist()
        if isinstance(v, (np.floating, np.integer)): return float(v)
        return v

    metrics_out = {k: _ser(v) for k, v in results.items()}
    with open("results/qaoa/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    with open("results/qaoa/selected_channels.json", "w") as f:
        json.dump({"retained": retained, "pruned": results["pruned_channels"],
                   "mask": mask.tolist()}, f, indent=2)

    qaoa_ser = {k: _ser(v) for k, v in qaoa_result.items()}
    with open("results/qaoa/solver_results.json", "w") as f:
        json.dump(qaoa_ser, f, indent=2)

    _print_results(results)
    return results, pruned_model


def _print_results(r: Dict[str, Any]) -> None:
    print(f"\n  QAOA RESULTS")
    print(f"  {'─'*40}")
    print(f"  Method             : {r.get('method')}")
    print(f"  Retained channels  : {r.get('retained_channels')}")
    print(f"  QAOA depth         : {r.get('qaoa_depth')}")
    print(f"  QAOA iterations    : {r.get('qaoa_iterations')}")
    print(f"  QAOA shots         : {r.get('qaoa_shots')}")
    print(f"  Clean Accuracy     : {r.get('accuracy', 'N/A'):.4f}")
    print(f"  Macro F1           : {r.get('f1', 'N/A'):.4f}")
    print(f"  QUBO Objective     : {r.get('qubo_objective', 'N/A'):.6f}")
    print(f"  Feasible           : {r.get('feasible')}")
    print(f"  Runtime (s)        : {r.get('runtime_s', 'N/A'):.2f}")
