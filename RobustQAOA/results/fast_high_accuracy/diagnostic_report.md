# Diagnostic Report — Fast High-Accuracy Baseline

Generated: 2026-10-02 20:25:07

## Dataset
- **CIFAR-10** (10 classes): airplane, automobile, bird, cat, deer,
  dog, frog, horse, ship, truck.
- Why CIFAR-10: CIFAR-100 with 10k images = 100 samples/class —
  two prior runs capped at ~40% accuracy on CPU.
  CIFAR-10 with 3k images (300/class) is tractable for transfer learning.
- Train: 3,000 | Val: 600 | Test: 600
- Stratified, fixed seed=42, no train/val/test overlap.

## Architecture: ResNet18Prunable
| Layer | Details |
|---|---|
| **conv1** (PRUNABLE) | Conv2d(3, **16**, 3, pad=1) |
| bn1 | BatchNorm2d(16) |
| proj | Conv2d(16, 64, 1) — restores ResNet width |
| bn_proj | BatchNorm2d(64) |
| layer1-4 | Pretrained ResNet18 residual blocks |
| avgpool | AdaptiveAvgPool2d(1,1) |
| fc | Linear(512, 10) |
| **Total params** | 11,173,722 |

## Training Configuration
| Phase 1 (frozen backbone) | 5 epochs, LR=0.01, SGD+Nesterov |
|---|---|
| Phase 2 (full fine-tune) | 20 epochs max, LR=0.05 |
| Scheduler | CosineAnnealingLR per phase |
| Early stopping | patience=7 epochs |
| Label smoothing | 0.1 |
| Augmentation | RandAugment(n=2,m=7) + RandomCrop + RandomFlip |
| Pretrained | ImageNet ResNet18 (torchvision) |
| Best epoch | 23 |
| Training time | 0s (0.0 min) |

## Test Results (fixed 600-image test subset — evaluated ONCE)
| Test Accuracy | **0.7967** |
|---|---|
| Macro Precision | 0.7952 |
| Macro Recall | 0.7967 |
| Macro F1 | 0.7953 |
| Test Loss | 0.6535 |
| Parameters | 11,173,722 |
| Memory (MB) | 42.624 |
| Inference (ms/sample) | 8.9759 |

## 70-80% Target: **REACHED**
Achieved: **79.67%**

## Pruning Compatibility
- `conv1` has exactly 16 output channels (verified).
- `forward_masked(x, mask)` applies mask between conv1 and proj layer.
- `build_pruned_model(model, retained_channels)` physically removes
  channels from conv1/bn1 and the corresponding input slice of proj.
- All downstream ResNet layers (layer1-4) are copied unchanged.
- Compatible with classical pruning, exact QUBO, and QAOA.

## Transfer Learning Note
Pretrained weights: ResNet18 trained on ImageNet (torchvision
ResNet18_Weights.IMAGENET1K_V1). Only layer1-layer4 + avgpool use
these weights. conv1, bn1, proj, bn_proj, and fc are randomly
initialised and trained from scratch.
