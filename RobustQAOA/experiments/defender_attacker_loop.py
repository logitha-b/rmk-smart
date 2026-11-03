"""
Iterative Defender-Attacker Loop.

Round structure:
    1. Defender: Build QUBO using current importance + robustness scores.
    2. Defender: Solve with classical method and QAOA.
    3. Defender: Construct pruned models.
    4. Attacker: Generate adversarial examples against the selected models.
    5. Feedback: Measure which retained channels contribute to robustness.
    6. Defender: Update robustness scores.
    7. Repeat for T rounds.

Records results after every round.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from optimization.qubo         import build_qubo_matrix, evaluate_qubo_vectorized
from optimization.exact_solver import exact_qubo_solve
from optimization.qaoa         import run_qaoa
from pruning.classical_pruning import classical_greedy_selection
from pruning.pruning_utils     import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
)
from models.pruned_model       import build_pruned_model
from training.evaluate         import evaluate_full
from robustness.attack_pool    import build_attack_pool
from robustness.robustness_score import (
    compute_robustness_scores,
    evaluate_model_robustness,
)


def run_defender_attacker_loop(
    baseline_model:   torch.nn.Module,
    importance:       np.ndarray,
    synergy:          np.ndarray,
    initial_robustness: np.ndarray,
    cfg:              Dict[str, Any],
    test_loader,
    device:           torch.device,
) -> pd.DataFrame:
    """
    Run the full iterative defender-attacker loop.

    Args:
        baseline_model      : trained SmallCNN (source of weights for pruning)
        importance          : c_i array, shape (16,)
        synergy             : Q_ij matrix, shape (16, 16)
        initial_robustness  : initial r_i array, shape (16,)
        cfg                 : config dict
        test_loader         : CIFAR-100 test DataLoader
        device              : torch device

    Returns:
        iteration_df : DataFrame recording metrics per round for both methods
    """
    T               = cfg.get("defender_attacker_rounds", 3)
    num_channels    = cfg.get("num_prunable_channels", 16)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)
    alpha           = cfg.get("alpha",         1.0)
    beta            = cfg.get("beta",          1.0)
    gamma           = cfg.get("gamma",         0.5)
    lam             = cfg.get("lambda_penalty", 10.0)
    seed            = cfg.get("seed",          42)

    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)

    robustness = initial_robustness.copy()

    records: List[Dict[str, Any]] = []

    print(f"\n{'='*60}")
    print(f"  DEFENDER-ATTACKER LOOP ({T} rounds)")
    print(f"{'='*60}")

    for t in range(1, T + 1):
        print(f"\n  ── Round {t}/{T} ──────────────────────────────────────")

        # ── Build QUBO ────────────────────────────────────────────────────────
        lin, quad = build_qubo_matrix(
            importance, robustness, synergy, w, M,
            alpha, beta, gamma, lam,
        )

        # ── Classical selection ───────────────────────────────────────────────
        c_mask, c_runtime, c_retained = classical_greedy_selection(
            importance, robustness, alpha, beta, num_channels, budget_fraction
        )
        c_obj = evaluate_qubo_vectorized(c_mask, lin, quad)

        # ── QAOA selection ────────────────────────────────────────────────────
        qaoa_result = run_qaoa(lin, quad, w, M, cfg)
        q_mask      = qaoa_result["best_mask"]
        q_retained  = qaoa_result["retained_channels"]
        q_runtime   = qaoa_result["runtime_s"]
        q_obj       = qaoa_result["best_objective"]

        # ── Build pruned models ───────────────────────────────────────────────
        c_model = build_pruned_model(baseline_model, c_retained, num_classes=100).to(device)
        q_model = build_pruned_model(baseline_model, q_retained, num_classes=100).to(device)

        # ── Evaluate clean accuracy ───────────────────────────────────────────
        c_clean = evaluate_full(c_model, test_loader, device)
        q_clean = evaluate_full(q_model, test_loader, device)

        # ── Generate attack pool against classical model ───────────────────────
        # (the attacker targets the currently deployed model)
        pool = build_attack_pool(c_model, test_loader, cfg, device)

        # ── Evaluate robustness ───────────────────────────────────────────────
        c_rob_metrics = evaluate_model_robustness(c_model, test_loader, cfg, device)
        q_rob_metrics = evaluate_model_robustness(q_model, test_loader, cfg, device)

        # ── Update robustness scores (feedback) ──────────────────────────────
        # Recompute r_i using the CLASSICAL pruned model as the defender
        # (we observe which retained channels help against the attack pool)
        new_robustness, _ = compute_robustness_scores(
            baseline_model, pool, device, num_channels
        )
        robustness = new_robustness
        print(f"  [round {t}] Updated robustness scores: {np.round(robustness, 4)}")

        # ── Record results ────────────────────────────────────────────────────
        records.append({
            "round":                    t,
            "classical_objective":      c_obj,
            "qaoa_objective":           q_obj,
            "classical_clean_accuracy": c_clean["accuracy"],
            "qaoa_clean_accuracy":      q_clean["accuracy"],
            "classical_robust_accuracy": c_rob_metrics["robust_accuracy"],
            "qaoa_robust_accuracy":     q_rob_metrics["robust_accuracy"],
            "classical_fgsm_accuracy":  c_rob_metrics["fgsm_accuracy"],
            "qaoa_fgsm_accuracy":       q_rob_metrics["fgsm_accuracy"],
            "classical_pgd_accuracy":   c_rob_metrics["pgd_accuracy"],
            "qaoa_pgd_accuracy":        q_rob_metrics["pgd_accuracy"],
            "classical_retained":       str(c_retained),
            "qaoa_retained":            str(q_retained),
            "classical_n_retained":     len(c_retained),
            "qaoa_n_retained":          len(q_retained),
            "classical_runtime_s":      c_runtime,
            "qaoa_runtime_s":           q_runtime,
        })

        print(f"  [round {t}] Classical: obj={c_obj:.4f} | acc={c_clean['accuracy']:.4f} | rob={c_rob_metrics['robust_accuracy']:.4f}")
        print(f"  [round {t}] QAOA:      obj={q_obj:.4f} | acc={q_clean['accuracy']:.4f} | rob={q_rob_metrics['robust_accuracy']:.4f}")

    # ── Save iterations ───────────────────────────────────────────────────────
    os.makedirs("results/iterations", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv("results/iterations/iterations.csv", index=False)
    print(f"\n  [loop] Iteration results saved to results/iterations/iterations.csv")

    return df
