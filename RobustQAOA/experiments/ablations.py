"""
Ablation studies.

Ablation 1: Without robustness term (beta=0)
Ablation 2: Without pairwise synergy (gamma=0)
Ablation 3: Without iterative attacker feedback (single round)
Ablation 4: Different memory budgets
Ablation 5: Different QAOA depths
Ablation 6: Different attack strengths (epsilon values)
"""

import os
import json
import copy
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

from optimization.qubo          import build_qubo_matrix, evaluate_qubo_vectorized
from optimization.exact_solver  import exact_qubo_solve
from optimization.qaoa          import run_qaoa
from pruning.classical_pruning  import classical_greedy_selection
from pruning.pruning_utils      import compute_channel_memory_costs, compute_memory_budget
from models.pruned_model        import build_pruned_model
from training.evaluate          import evaluate_full
from robustness.robustness_score import evaluate_model_robustness


def run_ablations(
    baseline_model:    torch.nn.Module,
    importance:        np.ndarray,
    robustness_scores: np.ndarray,
    synergy:           np.ndarray,
    cfg:               Dict[str, Any],
    test_loader,
    device:            torch.device,
) -> pd.DataFrame:
    """
    Run all ablation experiments.

    Returns a DataFrame with all ablation results.
    """
    records = []

    # ── Ablation 1: No robustness term ────────────────────────────────────────
    print("\n  [ablation 1] beta=0 (no robustness term)")
    cfg_abl = {**cfg, "beta": 0.0}
    r = _run_single_ablation("no_robustness", cfg_abl, baseline_model,
                              importance, robustness_scores, synergy, test_loader, device)
    records.append(r)

    # ── Ablation 2: No synergy ────────────────────────────────────────────────
    print("\n  [ablation 2] gamma=0 (no pairwise synergy)")
    cfg_abl = {**cfg, "gamma": 0.0}
    r = _run_single_ablation("no_synergy", cfg_abl, baseline_model,
                              importance, robustness_scores, synergy, test_loader, device)
    records.append(r)

    # ── Ablation 3: Single round (no iterative feedback) ─────────────────────
    print("\n  [ablation 3] Single round (no iterative feedback)")
    cfg_abl = {**cfg, "defender_attacker_rounds": 1}
    r = _run_single_ablation("single_round", cfg_abl, baseline_model,
                              importance, robustness_scores, synergy, test_loader, device)
    records.append(r)

    # ── Ablation 4: Different memory budgets ──────────────────────────────────
    for budget in [0.40, 0.50, 0.70, 0.80]:
        print(f"\n  [ablation 4] memory_budget_fraction={budget}")
        cfg_abl = {**cfg, "memory_budget_fraction": budget}
        r = _run_single_ablation(f"budget_{int(budget*100)}pct", cfg_abl, baseline_model,
                                  importance, robustness_scores, synergy, test_loader, device)
        records.append(r)

    # ── Ablation 5: Different QAOA depths ─────────────────────────────────────
    for depth in [1, 2, 3]:
        print(f"\n  [ablation 5] qaoa_depth={depth}")
        cfg_abl = {**cfg, "qaoa_depth": depth}
        r = _run_single_ablation(f"qaoa_depth_{depth}", cfg_abl, baseline_model,
                                  importance, robustness_scores, synergy, test_loader, device)
        records.append(r)

    # ── Ablation 6: Different attack strengths ────────────────────────────────
    for eps in [0.01, 0.03, 0.05]:
        print(f"\n  [ablation 6] fgsm_epsilon=pgd_epsilon={eps}")
        cfg_abl = {**cfg, "fgsm_epsilon": eps, "pgd_epsilon": eps}
        r = _run_single_ablation(f"epsilon_{int(eps*100):02d}", cfg_abl, baseline_model,
                                  importance, robustness_scores, synergy, test_loader, device)
        records.append(r)

    df = pd.DataFrame(records)

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/ablations.csv", index=False)
    print("\n  [ablation] Results saved to results/ablations.csv")

    return df


def _run_single_ablation(
    label:         str,
    cfg:           Dict[str, Any],
    baseline_model,
    importance:    np.ndarray,
    robustness:    np.ndarray,
    synergy:       np.ndarray,
    test_loader,
    device:        torch.device,
) -> Dict[str, Any]:
    """Run one ablation configuration and return a results row."""
    num_channels    = cfg.get("num_prunable_channels", 16)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)
    alpha           = cfg.get("alpha",         1.0)
    beta            = cfg.get("beta",          1.0)
    gamma           = cfg.get("gamma",         0.5)
    lam             = cfg.get("lambda_penalty", 10.0)

    w   = compute_channel_memory_costs(num_channels)
    M   = compute_memory_budget(num_channels, budget_fraction)
    lin, quad = build_qubo_matrix(importance, robustness, synergy, w, M,
                                   alpha, beta, gamma, lam)

    # Classical
    c_mask, c_runtime, c_retained = classical_greedy_selection(
        importance, robustness, alpha, beta, num_channels, budget_fraction
    )
    c_obj = evaluate_qubo_vectorized(c_mask, lin, quad)

    # QAOA
    qaoa_result = run_qaoa(lin, quad, w, M, cfg)
    q_retained  = qaoa_result["retained_channels"]
    q_obj       = qaoa_result["best_objective"]
    q_runtime   = qaoa_result["runtime_s"]

    # Build models and evaluate
    c_model = build_pruned_model(baseline_model, c_retained, num_classes=100).to(device)
    q_model = build_pruned_model(baseline_model, q_retained, num_classes=100).to(device)

    c_clean = evaluate_full(c_model, test_loader, device)
    q_clean = evaluate_full(q_model, test_loader, device)

    c_rob   = evaluate_model_robustness(c_model, test_loader, cfg, device)
    q_rob   = evaluate_model_robustness(q_model, test_loader, cfg, device)

    return {
        "ablation":                  label,
        "classical_objective":       c_obj,
        "qaoa_objective":            q_obj,
        "classical_clean_accuracy":  c_clean["accuracy"],
        "qaoa_clean_accuracy":       q_clean["accuracy"],
        "classical_robust_accuracy": c_rob["robust_accuracy"],
        "qaoa_robust_accuracy":      q_rob["robust_accuracy"],
        "classical_n_retained":      len(c_retained),
        "qaoa_n_retained":           len(q_retained),
        "classical_runtime_s":       c_runtime,
        "qaoa_runtime_s":            q_runtime,
        "memory_budget_fraction":    budget_fraction,
        "beta":                      beta,
        "gamma":                     gamma,
        "qaoa_depth":                cfg.get("qaoa_depth", 1),
        "fgsm_epsilon":              cfg.get("fgsm_epsilon", 0.03),
    }
