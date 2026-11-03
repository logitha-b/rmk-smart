"""
Automatic final report generator.

Reads from result dicts produced by all experiments and generates:
  results/final_report.md

All values come from actual experiment outputs.
No values are fabricated.
"""

import os
import json
import datetime
import numpy as np
from typing import Dict, Any, Optional


def generate_final_report(
    cfg:              Dict[str, Any],
    baseline_results: Dict[str, Any],
    classical_results: Dict[str, Any],
    exact_results:    Dict[str, Any],
    qaoa_results:     Dict[str, Any],
    integrity_report: Dict[str, Any],
    output_path:      str = "results/final_report.md",
) -> None:
    """Generate the final research report as a markdown file."""

    def _f(d, k, fmt=".4f"):
        v = d.get(k, None)
        if v is None:           return "N/A"
        if isinstance(v, list): return str(v)
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    def _diff(a, b, k, fmt=".4f"):
        va = a.get(k, None)
        vb = b.get(k, None)
        if va is None or vb is None:
            return "N/A"
        try:
            return format(float(vb) - float(va), fmt)
        except (TypeError, ValueError):
            return "N/A"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Final Research Report",
        f"*Generated: {now}*",
        "",
        "---",
        "",
        "## 1. Abstract",
        "",
        ("This report presents an experimental comparison of conventional classical "
         "channel pruning versus QAOA-based channel pruning for adversarially robust "
         "neural network compression on CIFAR-100. "
         "A small CNN with a 16-channel prunable bottleneck layer (conv1) is trained "
         "and pruned under a memory constraint while jointly optimizing for clean "
         "performance, adversarial robustness, and pairwise channel synergy. "
         "The optimization problem is formulated as a QUBO and solved by three methods: "
         "(1) classical greedy selection, (2) exact exhaustive enumeration (ground truth), "
         "and (3) QAOA via Qiskit Aer. "
         "Results are reported honestly — QAOA is not assumed to be superior."),
        "",
        "---",
        "",
        "## 2. Problem Definition",
        "",
        ("Channel pruning under memory constraint: given a trained CNN with 16 prunable "
         "channels, select a subset that maximizes a combined objective of clean accuracy "
         "contribution, adversarial robustness contribution, and pairwise synergy, "
         "while satisfying a memory budget constraint."),
        "",
        "Binary variables: x_i ∈ {0,1}, i = 0...15",
        "",
        ("Objective (maximize): "
         "α Σ c_i x_i + β Σ r_i x_i + γ Σ Q_ij x_i x_j - λ(Σ w_i x_i - M)²"),
        "",
        "---",
        "",
        "## 3. Methodology",
        "",
        "1. Train SmallCNN on CIFAR-100.",
        "2. Compute channel importance via single-channel ablation.",
        "3. Compute pairwise synergy via double-channel ablation.",
        "4. Build attack pool using FGSM and PGD.",
        "5. Compute per-channel robustness scores from the attack pool.",
        "6. Formulate QUBO.",
        "7. Solve with classical greedy, exact enumeration, and QAOA.",
        "8. Build physically pruned models for each solution.",
        "9. Evaluate all models.",
        "10. Run defender-attacker loop for iterative refinement.",
        "",
        "---",
        "",
        "## 4. Model Architecture",
        "",
        "```",
        "SmallCNN:",
        "  conv1: Conv2d(3, 16, 3, padding=1)   ← PRUNABLE BOTTLENECK",
        "  bn1:   BatchNorm2d(16)",
        "  conv2: Conv2d(16, 32, 3, padding=1)",
        "  bn2:   BatchNorm2d(32)",
        "  conv3: Conv2d(32, 64, 3, padding=1)",
        "  bn3:   BatchNorm2d(64)",
        "  fc1:   Linear(1024, 256)",
        "  fc2:   Linear(256, 100)",
        "```",
        "",
        f"Total parameters (baseline): {_f(baseline_results, 'parameters', ',d')}",
        "",
        "---",
        "",
        "## 5. Channel Importance",
        "",
        ("c_i = f(full model accuracy) - f(accuracy with channel i zeroed), "
         "evaluated on a fixed validation subset."),
        "",
        "---",
        "",
        "## 6. Pairwise Synergy",
        "",
        ("synergy_ij = effect_ij - effect_i - effect_j, where "
         "effect_i = f(full) - f(-i) and effect_ij = f(full) - f(-i,-j)."),
        "",
        "---",
        "",
        "## 7. QUBO Formulation",
        "",
        "maximize: α Σ c_i x_i + β Σ r_i x_i + γ Σ Q_ij x_i x_j - λ(Σ w_i x_i - M)²",
        "",
        f"- α (clean performance weight)   : {cfg.get('alpha', 'N/A')}",
        f"- β (robustness weight)          : {cfg.get('beta', 'N/A')}",
        f"- γ (synergy weight)             : {cfg.get('gamma', 'N/A')}",
        f"- λ (memory penalty)             : {cfg.get('lambda_penalty', 'N/A')}",
        f"- Memory budget fraction         : {cfg.get('memory_budget_fraction', 'N/A')}",
        "",
        "---",
        "",
        "## 8. Classical Optimization",
        "",
        "Greedy importance-based selection: sort channels by α·c_i + β·r_i, "
        "retain greedily while respecting memory budget.",
        "",
        f"- Retained channels: {classical_results.get('retained_channels', 'N/A')}",
        f"- Runtime: {_f(classical_results, 'runtime_s')} s",
        f"- QUBO Objective: {_f(classical_results, 'qubo_objective')}",
        "",
        "---",
        "",
        "## 9. QAOA",
        "",
        f"- QAOA depth (p): {qaoa_results.get('qaoa_depth', 'N/A')}",
        f"- Shots: {qaoa_results.get('qaoa_shots', 'N/A')}",
        f"- Optimizer: {qaoa_results.get('qaoa_optimizer', 'N/A')}",
        f"- Iterations: {qaoa_results.get('qaoa_iterations', 'N/A')}",
        f"- Method used: {qaoa_results.get('method', 'N/A')}",
        f"- Retained channels: {qaoa_results.get('retained_channels', 'N/A')}",
        f"- Runtime: {_f(qaoa_results, 'runtime_s')} s",
        f"- QUBO Objective: {_f(qaoa_results, 'qubo_objective')}",
        f"- Feasible: {qaoa_results.get('feasible', 'N/A')}",
        "",
        "---",
        "",
        "## 10. Defender-Attacker Loop",
        "",
        f"Rounds: {cfg.get('defender_attacker_rounds', 'N/A')}",
        "",
        ("In each round: the defender selects channels, the attacker generates "
         "adversarial examples, robustness scores are updated based on observed "
         "channel contributions, and the QUBO is rebuilt for the next round."),
        "",
        "---",
        "",
        "## 11. Experimental Setup",
        "",
        f"- Dataset: CIFAR-100",
        f"- Seed: {cfg.get('seed', 42)}",
        f"- Mode: {cfg.get('mode', 'N/A')}",
        f"- FGSM epsilon: {cfg.get('fgsm_epsilon', 'N/A')}",
        f"- PGD epsilon: {cfg.get('pgd_epsilon', 'N/A')}",
        f"- PGD step size: {cfg.get('pgd_step_size', 'N/A')}",
        f"- PGD steps: {cfg.get('pgd_steps', 'N/A')}",
        "",
        "---",
        "",
        "## 12. Results — Baseline Model",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Clean Accuracy | {_f(baseline_results, 'accuracy')} |",
        f"| Macro Precision | {_f(baseline_results, 'precision')} |",
        f"| Macro Recall | {_f(baseline_results, 'recall')} |",
        f"| Macro F1 | {_f(baseline_results, 'f1')} |",
        f"| Loss | {_f(baseline_results, 'loss')} |",
        f"| Parameters | {baseline_results.get('parameters', 'N/A')} |",
        f"| Memory (MB) | {_f(baseline_results, 'memory_mb')} |",
        f"| Inference Time (ms) | {_f(baseline_results, 'inference_ms')} |",
        "",
        "---",
        "",
        "## 13. Classical vs QAOA Comparison",
        "",
        "| Metric | Classical | QAOA | Difference (QAOA - Classical) |",
        "|--------|-----------|------|-------------------------------|",
        f"| Clean Accuracy | {_f(classical_results,'accuracy')} | {_f(qaoa_results,'accuracy')} | {_diff(classical_results, qaoa_results,'accuracy')} |",
        f"| Macro Precision | {_f(classical_results,'precision')} | {_f(qaoa_results,'precision')} | {_diff(classical_results, qaoa_results,'precision')} |",
        f"| Macro Recall | {_f(classical_results,'recall')} | {_f(qaoa_results,'recall')} | {_diff(classical_results, qaoa_results,'recall')} |",
        f"| Macro F1 | {_f(classical_results,'f1')} | {_f(qaoa_results,'f1')} | {_diff(classical_results, qaoa_results,'f1')} |",
        f"| Test Loss | {_f(classical_results,'loss')} | {_f(qaoa_results,'loss')} | {_diff(classical_results, qaoa_results,'loss')} |",
        f"| Parameters | {classical_results.get('parameters','N/A')} | {qaoa_results.get('parameters','N/A')} | — |",
        f"| Parameter Reduction % | {_f(classical_results,'param_reduction_pct')} | {_f(qaoa_results,'param_reduction_pct')} | {_diff(classical_results, qaoa_results,'param_reduction_pct')} |",
        f"| Memory (MB) | {_f(classical_results,'memory_mb')} | {_f(qaoa_results,'memory_mb')} | {_diff(classical_results, qaoa_results,'memory_mb')} |",
        f"| Memory Reduction % | {_f(classical_results,'mem_reduction_pct')} | {_f(qaoa_results,'mem_reduction_pct')} | {_diff(classical_results, qaoa_results,'mem_reduction_pct')} |",
        f"| Inference Time (ms) | {_f(classical_results,'inference_ms')} | {_f(qaoa_results,'inference_ms')} | {_diff(classical_results, qaoa_results,'inference_ms')} |",
        f"| FGSM Accuracy | {_f(classical_results,'fgsm_accuracy')} | {_f(qaoa_results,'fgsm_accuracy')} | {_diff(classical_results, qaoa_results,'fgsm_accuracy')} |",
        f"| PGD Accuracy | {_f(classical_results,'pgd_accuracy')} | {_f(qaoa_results,'pgd_accuracy')} | {_diff(classical_results, qaoa_results,'pgd_accuracy')} |",
        f"| Robust Accuracy | {_f(classical_results,'robust_accuracy')} | {_f(qaoa_results,'robust_accuracy')} | {_diff(classical_results, qaoa_results,'robust_accuracy')} |",
        f"| Worst Attack Loss | {_f(classical_results,'worst_attack_loss')} | {_f(qaoa_results,'worst_attack_loss')} | {_diff(classical_results, qaoa_results,'worst_attack_loss')} |",
        f"| Opt. Runtime (s) | {_f(classical_results,'runtime_s')} | {_f(qaoa_results,'runtime_s')} | {_diff(classical_results, qaoa_results,'runtime_s')} |",
        f"| QUBO Objective | {_f(classical_results,'qubo_objective')} | {_f(qaoa_results,'qubo_objective')} | {_diff(classical_results, qaoa_results,'qubo_objective')} |",
        "",
        "---",
        "",
        "## 14. QAOA vs Exact QUBO",
        "",
        "| Metric | Exact QUBO | QAOA | Difference (QAOA - Exact) |",
        "|--------|------------|------|---------------------------|",
        f"| Objective | {_f(exact_results,'qubo_objective')} | {_f(qaoa_results,'qubo_objective')} | {_diff(exact_results, qaoa_results,'qubo_objective')} |",
        f"| Runtime (s) | {_f(exact_results,'runtime_s')} | {_f(qaoa_results,'runtime_s')} | {_diff(exact_results, qaoa_results,'runtime_s')} |",
        "",
        _qaoa_optimum_found(exact_results, qaoa_results),
        "",
        "---",
        "",
        "## 15. Adversarial Robustness",
        "",
        "| Model | Clean Acc | FGSM Acc | PGD Acc | Robust Acc |",
        "|-------|-----------|----------|---------|------------|",
        f"| Classical | {_f(classical_results,'accuracy')} | {_f(classical_results,'fgsm_accuracy')} | {_f(classical_results,'pgd_accuracy')} | {_f(classical_results,'robust_accuracy')} |",
        f"| Exact QUBO | {_f(exact_results,'accuracy')} | {_f(exact_results,'fgsm_accuracy')} | {_f(exact_results,'pgd_accuracy')} | {_f(exact_results,'robust_accuracy')} |",
        f"| QAOA | {_f(qaoa_results,'accuracy')} | {_f(qaoa_results,'fgsm_accuracy')} | {_f(qaoa_results,'pgd_accuracy')} | {_f(qaoa_results,'robust_accuracy')} |",
        "",
        "---",
        "",
        "## 16. Compression Results",
        "",
        "| Model | Parameters | Param Reduction % | Memory (MB) | Mem Reduction % |",
        "|-------|------------|-------------------|-------------|-----------------|",
        f"| Classical | {classical_results.get('parameters','N/A')} | {_f(classical_results,'param_reduction_pct')} | {_f(classical_results,'memory_mb')} | {_f(classical_results,'mem_reduction_pct')} |",
        f"| Exact QUBO | {exact_results.get('parameters','N/A')} | {_f(exact_results,'param_reduction_pct')} | {_f(exact_results,'memory_mb')} | {_f(exact_results,'mem_reduction_pct')} |",
        f"| QAOA | {qaoa_results.get('parameters','N/A')} | {_f(qaoa_results,'param_reduction_pct')} | {_f(qaoa_results,'memory_mb')} | {_f(qaoa_results,'mem_reduction_pct')} |",
        "",
        "---",
        "",
        "## 17. Integrity Check",
        "",
        _integrity_section(integrity_report),
        "",
        "---",
        "",
        "## 18. Limitations",
        "",
        "- CIFAR-100 is a standard benchmark; results may not generalize to other domains.",
        "- QAOA depth p is small due to simulator computational constraints.",
        "- Robustness is empirical (FGSM/PGD); no formal certification is provided.",
        "- Exhaustive verification of the full CIFAR-100 image space is infeasible.",
        "- The 16-channel bottleneck is intentionally small for tractability.",
        "",
        "---",
        "",
        "## 19. Conclusion",
        "",
        ("This experiment measured the difference between classical greedy channel "
         "pruning and QAOA-based channel pruning on CIFAR-100 under identical "
         "experimental conditions. "
         f"The QUBO objective difference (QAOA - Classical) was "
         f"{_diff(classical_results, qaoa_results, 'qubo_objective')}. "
         f"The clean accuracy difference was "
         f"{_diff(classical_results, qaoa_results, 'accuracy')}. "
         f"The robust accuracy difference was "
         f"{_diff(classical_results, qaoa_results, 'robust_accuracy')}. "
         "No a-priori claims about quantum advantage were made; the results reflect "
         "the actual experimental outcome."),
        "",
        "---",
        "*End of report*",
    ]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [report] Final report saved to {output_path}")


def _qaoa_optimum_found(exact, qaoa) -> str:
    em = exact.get("best_mask", exact.get("mask", None))
    qm = qaoa.get("best_mask",  qaoa.get("mask",  None))
    if em is None or qm is None:
        return "Exact optimum found by QAOA: UNKNOWN"
    em_list = np.array(em).astype(int).tolist() if not isinstance(em, list) else em
    qm_list = np.array(qm).astype(int).tolist() if not isinstance(qm, list) else qm
    found   = "YES" if em_list == qm_list else "NO"
    return f"**Exact optimum found by QAOA: {found}**"


def _integrity_section(ir: Dict[str, Any]) -> str:
    if not ir:
        return "Integrity check not available."
    lines = []
    for k, v in ir.items():
        status = "✓" if v is True else ("✗ FAILED" if v is False else str(v))
        lines.append(f"- [{status}] {k}")
    return "\n".join(lines)
