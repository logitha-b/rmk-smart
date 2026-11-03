# Final Research Report
*Generated: 2026-10-01 19:31:35*

---

## 1. Abstract

This report presents an experimental comparison of conventional classical channel pruning versus QAOA-based channel pruning for adversarially robust neural network compression on CIFAR-100. A small CNN with a 16-channel prunable bottleneck layer (conv1) is trained and pruned under a memory constraint while jointly optimizing for clean performance, adversarial robustness, and pairwise channel synergy. The optimization problem is formulated as a QUBO and solved by three methods: (1) classical greedy selection, (2) exact exhaustive enumeration (ground truth), and (3) QAOA via Qiskit Aer. Results are reported honestly — QAOA is not assumed to be superior.

---

## 2. Problem Definition

Channel pruning under memory constraint: given a trained CNN with 16 prunable channels, select a subset that maximizes a combined objective of clean accuracy contribution, adversarial robustness contribution, and pairwise synergy, while satisfying a memory budget constraint.

Binary variables: x_i ∈ {0,1}, i = 0...15

Objective (maximize): α Σ c_i x_i + β Σ r_i x_i + γ Σ Q_ij x_i x_j - λ(Σ w_i x_i - M)²

---

## 3. Methodology

1. Train SmallCNN on CIFAR-100.
2. Compute channel importance via single-channel ablation.
3. Compute pairwise synergy via double-channel ablation.
4. Build attack pool using FGSM and PGD.
5. Compute per-channel robustness scores from the attack pool.
6. Formulate QUBO.
7. Solve with classical greedy, exact enumeration, and QAOA.
8. Build physically pruned models for each solution.
9. Evaluate all models.
10. Run defender-attacker loop for iterative refinement.

---

## 4. Model Architecture

```
SmallCNN:
  conv1: Conv2d(3, 16, 3, padding=1)   ← PRUNABLE BOTTLENECK
  bn1:   BatchNorm2d(16)
  conv2: Conv2d(16, 32, 3, padding=1)
  bn2:   BatchNorm2d(32)
  conv3: Conv2d(32, 64, 3, padding=1)
  bn3:   BatchNorm2d(64)
  fc1:   Linear(1024, 256)
  fc2:   Linear(256, 100)
```

Total parameters (baseline): 311796

---

## 5. Channel Importance

c_i = f(full model accuracy) - f(accuracy with channel i zeroed), evaluated on a fixed validation subset.

---

## 6. Pairwise Synergy

synergy_ij = effect_ij - effect_i - effect_j, where effect_i = f(full) - f(-i) and effect_ij = f(full) - f(-i,-j).

---

## 7. QUBO Formulation

maximize: α Σ c_i x_i + β Σ r_i x_i + γ Σ Q_ij x_i x_j - λ(Σ w_i x_i - M)²

- α (clean performance weight)   : 1.0
- β (robustness weight)          : 1.0
- γ (synergy weight)             : 0.5
- λ (memory penalty)             : 5.0
- Memory budget fraction         : 0.6

---

## 8. Classical Optimization

Greedy importance-based selection: sort channels by α·c_i + β·r_i, retain greedily while respecting memory budget.

- Retained channels: [1, 3, 4, 5, 6, 9, 11, 13, 14]
- Runtime: 0.0036 s
- QUBO Objective: 1.8600

---

## 9. QAOA

- QAOA depth (p): 1
- Shots: 512
- Optimizer: COBYLA
- Iterations: 31
- Method used: qaoa
- Retained channels: [0, 1, 6, 9, 10, 12, 13, 14, 15]
- Runtime: 6.4113 s
- QUBO Objective: 747332784.6925
- Feasible: True

---

## 10. Defender-Attacker Loop

Rounds: 2

In each round: the defender selects channels, the attacker generates adversarial examples, robustness scores are updated based on observed channel contributions, and the QUBO is rebuilt for the next round.

---

## 11. Experimental Setup

- Dataset: CIFAR-100
- Seed: 42
- Mode: debug
- FGSM epsilon: 0.03
- PGD epsilon: 0.03
- PGD step size: 0.007
- PGD steps: 10

---

## 12. Results — Baseline Model

| Metric | Value |
|--------|-------|
| Clean Accuracy | 0.4520 |
| Macro Precision | 0.4366 |
| Macro Recall | 0.4370 |
| Macro F1 | 0.3969 |
| Loss | 2.0566 |
| Parameters | 311796 |
| Memory (MB) | 1.1894 |
| Inference Time (ms) | 0.5777 |

---

## 13. Classical vs QAOA Comparison

| Metric | Classical | QAOA | Difference (QAOA - Classical) |
|--------|-----------|------|-------------------------------|
| Clean Accuracy | 0.1300 | 0.0200 | -0.1100 |
| Macro Precision | 0.1867 | 0.0169 | -0.1698 |
| Macro Recall | 0.1372 | 0.0223 | -0.1149 |
| Macro F1 | 0.1246 | 0.0111 | -0.1135 |
| Test Loss | 4.2588 | 11.0797 | 6.8209 |
| Parameters | 309577 | 309577 | — |
| Parameter Reduction % | 0.7117 | 0.7117 | 0.0000 |
| Memory (MB) | 1.1809 | 1.1809 | 0.0000 |
| Memory Reduction % | 0.7117 | 0.7117 | 0.0000 |
| Inference Time (ms) | 1.8271 | 1.6765 | -0.1506 |
| FGSM Accuracy | 0.0391 | 0.0234 | -0.0156 |
| PGD Accuracy | 0.0234 | 0.0234 | 0.0000 |
| Robust Accuracy | 0.0234 | 0.0234 | 0.0000 |
| Worst Attack Loss | 7.0239 | 14.1741 | 7.1501 |
| Opt. Runtime (s) | 0.0036 | 6.4113 | 6.4077 |
| QUBO Objective | 1.8600 | 747332784.6925 | 747332782.8325 |

---

## 14. QAOA vs Exact QUBO

| Metric | Exact QUBO | QAOA | Difference (QAOA - Exact) |
|--------|------------|------|---------------------------|
| Objective | 747332784.7650 | 747332784.6925 | -0.0725 |
| Runtime (s) | 0.3223 | 6.4113 | 6.0890 |

**Exact optimum found by QAOA: NO**

---

## 15. Adversarial Robustness

| Model | Clean Acc | FGSM Acc | PGD Acc | Robust Acc |
|-------|-----------|----------|---------|------------|
| Classical | 0.1300 | 0.0391 | 0.0234 | 0.0234 |
| Exact QUBO | 0.0180 | 0.0117 | 0.0117 | 0.0117 |
| QAOA | 0.0200 | 0.0234 | 0.0234 | 0.0234 |

---

## 16. Compression Results

| Model | Parameters | Param Reduction % | Memory (MB) | Mem Reduction % |
|-------|------------|-------------------|-------------|-----------------|
| Classical | 309577 | 0.7117 | 1.1809 | 0.7117 |
| Exact QUBO | 309577 | 0.7117 | 1.1809 | 0.7117 |
| QAOA | 309577 | 0.7117 | 1.1809 | 0.7117 |

---

## 17. Integrity Check

- [✓] baseline_model_trained
- [✓] baseline_metrics_computed
- [✓] same_baseline_checkpoint_used
- [✓] classical_pruning_executed
- [✓] exact_qubo_executed
- [✓] qaoa_executed
- [✓] actual_channel_pruning_done
- [✓] memory_budget_satisfied_classical
- [✓] memory_budget_satisfied_qaoa
- [✓] fgsm_executed
- [✓] pgd_executed
- [✓] all_metrics_from_real_outputs
- [✓] no_nan_metrics

---

## 18. Limitations

- CIFAR-100 is a standard benchmark; results may not generalize to other domains.
- QAOA depth p is small due to simulator computational constraints.
- Robustness is empirical (FGSM/PGD); no formal certification is provided.
- Exhaustive verification of the full CIFAR-100 image space is infeasible.
- The 16-channel bottleneck is intentionally small for tractability.

---

## 19. Conclusion

This experiment measured the difference between classical greedy channel pruning and QAOA-based channel pruning on CIFAR-100 under identical experimental conditions. The QUBO objective difference (QAOA - Classical) was 747332782.8325. The clean accuracy difference was -0.1100. The robust accuracy difference was 0.0000. No a-priori claims about quantum advantage were made; the results reflect the actual experimental outcome.

---
*End of report*