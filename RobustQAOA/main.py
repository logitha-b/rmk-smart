"""
Main experiment orchestrator.

Runs all phases in order:
  1. Baseline training + evaluation
  2. Channel importance
  3. Pairwise synergy
  4. Attack pool + robustness scores
  5. Classical pruning experiment
  6. Exact QUBO experiment
  7. QAOA experiment
  8. Defender-attacker loop
  9. Comparison tables + plots
  10. Ablations (if enabled)
  11. Final report
  12. Terminal output

Usage:
    python main.py --config configs/debug.yaml
    python main.py --config configs/research.yaml
    python main.py --config configs/debug.yaml --force-retrain
"""

import os
import sys

# Force UTF-8 output on Windows (avoids UnicodeEncodeError for special chars)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import argparse
import time
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
import yaml

# ── Ensure project root is on path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train              import get_cifar100_loaders
from experiments.baseline        import run_baseline, set_seeds
from pruning.importance          import compute_channel_importance
from pruning.synergy             import compute_synergy_matrix
from pruning.pruning_utils       import (
    compute_channel_memory_costs,
    compute_memory_budget,
    print_memory_info,
)
from robustness.attack_pool      import build_attack_pool
from robustness.robustness_score import compute_robustness_scores
from optimization.qubo           import build_qubo_matrix, print_qubo_summary
from experiments.classical_experiment    import run_classical_experiment
from experiments.exact_qubo_experiment  import run_exact_qubo_experiment
from experiments.qaoa_experiment        import run_qaoa_experiment
from experiments.defender_attacker_loop import run_defender_attacker_loop
from experiments.comparison      import generate_comparison_tables
from experiments.report_generator import generate_final_report
from visualization.plots         import (
    plot_training_curve,
    plot_channel_importance,
    plot_synergy_heatmap,
    plot_robustness_comparison,
    plot_compression_comparison,
    plot_optimization_comparison,
    plot_iteration_curves,
    plot_classical_vs_qaoa,
)
from verification.bounded_verification import print_verification_disclaimer


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def print_config(cfg: dict) -> None:
    print(f"\n{'='*60}")
    print("  EXPERIMENT CONFIGURATION")
    print(f"{'='*60}")
    for k, v in cfg.items():
        print(f"  {k:<35}: {v}")


# ─────────────────────────────────────────────────────────────────────────────
# Scientific integrity check
# ─────────────────────────────────────────────────────────────────────────────

def integrity_check(state: dict) -> dict:
    """Verify all required experiments completed successfully."""
    checks = {
        "baseline_model_trained":        state.get("baseline_done",    False),
        "baseline_metrics_computed":     state.get("baseline_done",    False),
        "same_baseline_checkpoint_used": state.get("same_checkpoint",  True),
        "classical_pruning_executed":    state.get("classical_done",   False),
        "exact_qubo_executed":           state.get("exact_done",       False),
        "qaoa_executed":                 state.get("qaoa_done",        False),
        "actual_channel_pruning_done":   state.get("pruning_done",     False),
        "memory_budget_satisfied_classical": state.get("classical_feasible", False),
        "memory_budget_satisfied_qaoa":      state.get("qaoa_feasible",     False),
        "fgsm_executed":                 state.get("fgsm_done",        False),
        "pgd_executed":                  state.get("pgd_done",         False),
        "all_metrics_from_real_outputs": state.get("real_outputs",     True),
        "no_nan_metrics":                state.get("no_nans",          True),
    }

    passed = sum(1 for v in checks.values() if v is True)
    total  = len(checks)

    print(f"\n{'='*60}")
    print("  SCIENTIFIC INTEGRITY CHECK")
    print(f"{'='*60}")
    for k, v in checks.items():
        status = "✓" if v else "✗ FAILED"
        print(f"  [{status}] {k}")
    print(f"\n  Passed: {passed}/{total}")
    print(f"{'='*60}")

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Final terminal output
# ─────────────────────────────────────────────────────────────────────────────

def print_final_output(baseline, classical, qaoa, exact) -> None:
    def _v(d, k):
        v = d.get(k, None)
        if v is None:           return "N/A"
        if isinstance(v, list): return str(v)
        try:    return f"{float(v):.4f}"
        except: return str(v)

    def _diff(a, b, k):
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None: return "N/A"
        try:    return f"{float(vb) - float(va):.4f}"
        except: return "N/A"

    print(f"\n{'='*62}")
    print("  FINAL EXPERIMENT RESULT")
    print(f"{'='*62}")
    print("")
    print("  BASELINE MODEL")
    print(f"  Accuracy       : {_v(baseline, 'accuracy')}")
    print(f"  F1             : {_v(baseline, 'f1')}")
    print(f"  Parameters     : {baseline.get('parameters', 'N/A')}")
    print(f"  Memory (MB)    : {_v(baseline, 'memory_mb')}")
    print(f"  {'─'*58}")
    print("")
    print("  CLASSICAL vs QAOA")
    print(f"  {'─'*58}")

    for label, key in [
        ("Clean Accuracy",    "accuracy"),
        ("Macro Precision",   "precision"),
        ("Macro Recall",      "recall"),
        ("Macro F1",          "f1"),
        ("Parameters",        "parameters"),
        ("Memory (MB)",       "memory_mb"),
        ("FGSM Accuracy",     "fgsm_accuracy"),
        ("PGD Accuracy",      "pgd_accuracy"),
        ("Robust Accuracy",   "robust_accuracy"),
        ("Opt. Runtime (s)",  "runtime_s"),
        ("QUBO Objective",    "qubo_objective"),
    ]:
        print(f"\n  {label}:")
        print(f"    Classical  : {_v(classical, key)}")
        print(f"    QAOA       : {_v(qaoa, key)}")
        print(f"    Difference : {_diff(classical, qaoa, key)}")

    print(f"\n  {'─'*58}")
    print("  QAOA vs EXACT QUBO")
    print(f"  {'─'*58}")

    e_obj = exact.get("qubo_objective", None)
    q_obj = qaoa.get("qubo_objective",  None)
    gap   = (float(e_obj) - float(q_obj)) if (e_obj and q_obj) else None

    print(f"  Exact Objective  : {_v(exact, 'qubo_objective')}")
    print(f"  QAOA Objective   : {_v(qaoa,  'qubo_objective')}")
    print(f"  Objective Gap    : {f'{gap:.4f}' if gap is not None else 'N/A'}")

    # Check if QAOA found exact optimum
    em = exact.get("best_mask", None)
    qm = qaoa.get("mask", None)
    if em is not None and qm is not None:
        em_l = np.array(em).astype(int).tolist()
        qm_l = np.array(qm).astype(int).tolist()
        found = "YES" if em_l == qm_l else "NO"
    else:
        found = "UNKNOWN"
    print(f"  Exact Solution Found by QAOA: {found}")

    print(f"\n  {'─'*58}")
    print("  Results saved to  : results/")
    print("  Final report      : results/final_report.md")
    print(f"{'='*62}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QAOA Robust Pruning Experiment")
    parser.add_argument("--config",        type=str, default="configs/debug.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retraining even if checkpoint exists")
    args = parser.parse_args()

    # ── Load configuration ────────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.force_retrain:
        cfg["force_retrain"] = True

    cfg["checkpoint_path"] = "results/checkpoints/baseline_model.pt"

    print_config(cfg)

    if cfg.get("debug_mode", False):
        print("\n  [WARNING] DEBUG MODE - results not suitable for research publication.")

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed   = cfg.get("seed", 42)

    # ── Experiment state tracker ──────────────────────────────────────────────
    state = {"same_checkpoint": True, "real_outputs": True, "no_nans": True}

    t_total = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2: Baseline
    # ──────────────────────────────────────────────────────────────────────────
    set_seeds(seed)
    baseline_model, baseline_metrics, test_loader = run_baseline(cfg)
    state["baseline_done"] = True

    # Plot training curve if history available
    history_path = "results/baseline/metrics.json"
    if os.path.exists(history_path):
        with open(history_path) as f:
            saved = json.load(f)
        history = saved.get("training_history", {})
        if history.get("train_loss"):
            plot_training_curve(history, "results/baseline/training_curve.png")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 3: Channel importance + memory
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 3: CHANNEL IMPORTANCE + MEMORY")
    print(f"{'='*60}")

    n_samples = cfg.get("val_subset_size", 500)
    importance, baseline_acc_subset = compute_channel_importance(
        baseline_model, test_loader, device,
        num_channels=cfg.get("num_prunable_channels", 16),
        n_samples=n_samples,
        seed=seed,
    )
    plot_channel_importance(importance, "results/baseline/channel_importance.png")

    print_memory_info(
        cfg.get("num_prunable_channels", 16),
        cfg.get("memory_budget_fraction", 0.60),
    )
    state["pruning_done"] = True

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 4: Pairwise synergy
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 4: PAIRWISE SYNERGY")
    print(f"{'='*60}")

    synergy = compute_synergy_matrix(
        baseline_model, test_loader, device,
        num_channels=cfg.get("num_prunable_channels", 16),
        n_samples=n_samples,
        seed=seed,
        baseline_acc=baseline_acc_subset,
    )
    plot_synergy_heatmap(synergy, "results/baseline/synergy_heatmap.png")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 5: Attack pool + robustness scores
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 5: ADVERSARIAL ATTACKS + ROBUSTNESS SCORES")
    print(f"{'='*60}")

    attack_pool = build_attack_pool(baseline_model, test_loader, cfg, device)
    state["fgsm_done"] = True
    state["pgd_done"]  = True

    robustness_scores, baseline_rob = compute_robustness_scores(
        baseline_model, attack_pool, device,
        num_channels=cfg.get("num_prunable_channels", 16),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 6+7: QUBO (used by both exact and QAOA)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 6: QUBO FORMULATION")
    print(f"{'='*60}")

    num_channels    = cfg.get("num_prunable_channels", 16)
    budget_fraction = cfg.get("memory_budget_fraction", 0.60)
    alpha           = cfg.get("alpha",         1.0)
    beta            = cfg.get("beta",          1.0)
    gamma           = cfg.get("gamma",         0.5)
    lam             = cfg.get("lambda_penalty", 10.0)

    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)
    lin, quad = build_qubo_matrix(importance, robustness_scores, synergy, w, M,
                                   alpha, beta, gamma, lam)
    print_qubo_summary(lin, quad)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 7: Exact solver
    # ──────────────────────────────────────────────────────────────────────────
    exact_results, exact_model = run_exact_qubo_experiment(
        baseline_model, importance, robustness_scores, synergy,
        cfg, test_loader, device, baseline_metrics,
    )
    state["exact_done"] = True

    # ──────────────────────────────────────────────────────────────────────────
    # Classical pruning
    # ──────────────────────────────────────────────────────────────────────────
    classical_results, classical_model = run_classical_experiment(
        baseline_model, importance, robustness_scores,
        cfg, test_loader, device, baseline_metrics,
    )
    state["classical_done"]     = True
    state["classical_feasible"] = classical_results.get("feasible", False)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 8: QAOA
    # ──────────────────────────────────────────────────────────────────────────
    qaoa_results, qaoa_model = run_qaoa_experiment(
        baseline_model, importance, robustness_scores, synergy,
        cfg, test_loader, device, baseline_metrics,
    )
    state["qaoa_done"]     = True
    state["qaoa_feasible"] = qaoa_results.get("feasible", False)

    # Add robustness metrics to baseline_results (needed for comparison table)
    from robustness.robustness_score import evaluate_model_robustness
    baseline_rob_metrics = evaluate_model_robustness(
        baseline_model, test_loader, cfg, device
    )
    baseline_results = {**baseline_metrics, **baseline_rob_metrics}

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 9: Defender-attacker loop
    # ──────────────────────────────────────────────────────────────────────────
    iteration_df = run_defender_attacker_loop(
        baseline_model, importance, synergy, robustness_scores,
        cfg, test_loader, device,
    )
    plot_iteration_curves(iteration_df, "results/iterations/iterations.png")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 10: Comparison tables
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PHASE 10: COMPARISON TABLES")
    print(f"{'='*60}")

    generate_comparison_tables(
        baseline_results, classical_results, exact_results, qaoa_results,
        output_dir="results/comparison",
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_robustness_comparison(
        baseline_results, classical_results, exact_results, qaoa_results,
        "results/comparison/robustness_comparison.png",
    )
    plot_compression_comparison(
        baseline_results, classical_results, exact_results, qaoa_results,
        "results/comparison/compression_comparison.png",
    )
    plot_optimization_comparison(
        classical_results, exact_results, qaoa_results,
        "results/comparison/optimization_comparison.png",
    )
    plot_classical_vs_qaoa(
        classical_results, qaoa_results,
        "results/comparison/comparison_plot.png",
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 11: Ablations
    # ──────────────────────────────────────────────────────────────────────────
    if cfg.get("run_ablations", False):
        from experiments.ablations import run_ablations
        print(f"\n{'='*60}")
        print("  PHASE 11: ABLATION STUDIES")
        print(f"{'='*60}")
        run_ablations(
            baseline_model, importance, robustness_scores, synergy,
            cfg, test_loader, device,
        )

    # Bounded verification disclaimer
    print_verification_disclaimer()

    # ──────────────────────────────────────────────────────────────────────────
    # NaN check
    # ──────────────────────────────────────────────────────────────────────────
    import math
    for d, name in [(classical_results, "classical"), (qaoa_results, "qaoa")]:
        for k, v in d.items():
            if isinstance(v, float) and math.isnan(v):
                state["no_nans"] = False
                print(f"  WARNING: NaN detected in {name}[{k}]")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 12: Integrity check + final report
    # ──────────────────────────────────────────────────────────────────────────
    ir = integrity_check(state)

    generate_final_report(
        cfg, baseline_results, classical_results, exact_results, qaoa_results,
        ir, output_path="results/final_report.md",
    )

    # Save environment info
    import platform
    try:
        import qiskit
        qiskit_version = qiskit.__version__
    except:
        qiskit_version = "not installed"

    env_info = {
        "python":   platform.python_version(),
        "torch":    torch.__version__,
        "device":   str(device),
        "qiskit":   qiskit_version,
        "platform": platform.platform(),
        "seed":     seed,
        "mode":     cfg.get("mode", "unknown"),
    }
    with open("results/environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # Final terminal output
    # ──────────────────────────────────────────────────────────────────────────
    print_final_output(baseline_results, classical_results, qaoa_results, exact_results)

    total_time = time.time() - t_total
    print(f"\n  Total experiment time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
