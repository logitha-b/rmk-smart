# CLASSICAL vs QAOA EXPERIMENTAL COMPARISON

## Main Comparison Table

| Metric                   | Classical / Actual Output | QAOA Output     | Difference (QAOA - Classical) |
|--------------------------|---------------------------|-----------------|-------------------------------|
| Clean Accuracy           | 0.1517                    | 0.0370          | -0.1147                       |
| Macro Precision          | 0.3250                    | 0.0791          | -0.2459                       |
| Macro Recall             | 0.1517                    | 0.0370          | -0.1147                       |
| Macro F1                 | 0.1482                    | 0.0234          | -0.1248                       |
| Test Loss                | 4.2424                    | 8.3693          | 4.1270                        |
| Parameters               | 309577                    | 309577          | 0.0000                        |
| Parameter Reduction %    | 0.7117                    | 0.7117          | 0.0000                        |
| Memory (MB)              | 1.1809                    | 1.1809          | 0.0000                        |
| Memory Reduction %       | 0.7117                    | 0.7117          | 0.0000                        |
| Inference Time (ms)      | 1.1262                    | 0.6788          | -0.4474                       |
| FGSM Accuracy            | 0.0366                    | 0.0239          | -0.0127                       |
| PGD Accuracy             | 0.0239                    | 0.0200          | -0.0039                       |
| Robust Accuracy          | 0.0239                    | 0.0200          | -0.0039                       |
| Worst Attack Loss        | 7.1000                    | 10.7374         | 3.6375                        |
| Retained Channels        | 9                         | 9               | 0                             |
| Pruned Channels          | 7                         | 7               | 0                             |
| Optimization Runtime (s) | 0.0040                    | 12.0191         | 12.0151                       |
| QUBO Objective           | 1.5510                    | 1494665568.6295 | 1494665567.0785               |

## QAOA vs Exact QUBO

| Metric              | Exact QUBO      | QAOA            | Difference |
|---------------------|-----------------|-----------------|------------|
| Objective           | 1494665568.6350 | 1494665568.6295 | -0.0055    |
| Retained Channels   | 9               | 9               | 0          |
| Pruned Channels     | 7               | 7               | 0          |
| Runtime (s)         | 0.3743          | 12.0191         | 11.6448    |
| Exact Optimum Found | YES             | UNKNOWN         | —          |

## Adversarial Robustness Comparison

| Model      | Clean Accuracy | FGSM Accuracy | PGD Accuracy | Robust Accuracy |
|------------|----------------|---------------|--------------|-----------------|
| Original   | 0.4233         | 0.1504        | 0.1035       | 0.1035          |
| Classical  | 0.1517         | 0.0366        | 0.0239       | 0.0239          |
| Exact QUBO | 0.0236         | 0.0186        | 0.0186       | 0.0186          |
| QAOA       | 0.0370         | 0.0239        | 0.0200       | 0.0200          |

## Compression Comparison

| Model      | Parameters | Parameter Reduction % | Memory (MB) | Memory Reduction % |
|------------|------------|-----------------------|-------------|--------------------|
| Original   | 311796     | 0.0000                | 1.1894      | 0.0000             |
| Classical  | 309577     | 0.7117                | 1.1809      | 0.7117             |
| Exact QUBO | 309577     | 0.7117                | 1.1809      | 0.7117             |
| QAOA       | 309577     | 0.7117                | 1.1809      | 0.7117             |

## Optimization Comparison

| Method     | Objective       | Runtime (s) | Feasible | Retained Channels |
|------------|-----------------|-------------|----------|-------------------|
| Classical  | 1.5510          | 0.0040      | True     | 9                 |
| Exact QUBO | 1494665568.6350 | 0.3743      | True     | 9                 |
| QAOA       | 1494665568.6295 | 12.0191     | True     | 9                 |

---
*Retained channels — Classical: [1, 3, 4, 5, 6, 9, 11, 13, 14]*
*Retained channels — QAOA: [0, 1, 2, 5, 8, 10, 12, 13, 15]*
*Retained channels — Exact QUBO: [0, 1, 4, 5, 8, 10, 11, 12, 15]*