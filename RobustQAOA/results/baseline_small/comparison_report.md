# Baseline Comparison

| Metric | Existing Baseline | Small-Data Improved | Difference |
|---|---:|---:|---:|
| Test Accuracy | 0.4233 | 0.4025 | -0.0208 |
| Macro Precision | 0.4511 | 0.4014 | -0.0497 |
| Macro Recall | 0.4233 | 0.4025 | -0.0208 |
| Macro F1 | 0.4120 | 0.3935 | -0.0184 |
| Test Loss | 2.1796 | 2.3768 | +0.1973 |
| Parameters | 311,796 | 1,184,180 | +872,384 |
| Memory (MB) | 1.1894 | 4.5173 | +3.3279 |
| Inference (ms) | 9.8418 | 0.6992 | -9.1426 |
| Training Time (s) | N/A | 533.0 | N/A |

> **NOTE (comparability):** Existing baseline was evaluated on 10,000 test samples (full CIFAR-100 test set). New baseline uses a fixed stratified subset of 2,000 test images from the official test split. Accuracy figures are **not directly comparable** due to different test-set sizes.