# Diagnostic Report — Small-Dataset Improved Baseline

Generated: 2026-10-02 14:13:42

## Dataset
- Source: CIFAR-100 (official splits)
- Train subset: 10,000 images (stratified, seed=42)
- Val subset:   2,000   images (stratified, from official train, no overlap)
- Test subset:  2,000   images (stratified, from official test split)
- Classes: 100

## Architecture Changes vs. Existing Baseline
| Component | Original SmallCNN | ImprovedSmallCNN |
|---|---|---|
| conv1 (prunable) | Conv2d(3→16) | Conv2d(3→16) **unchanged** |
| conv2 | Conv2d(16→32) | Conv2d(16→64) |
| conv3 | Conv2d(32→64) | Conv2d(64→128) |
| fc1   | Linear(1024→256) | Linear(2048→512) |
| Dropout | 0.3 | 0.4 |

## Training Configuration
| Optimizer | AdamW (lr=0.001, wd=0.01) |
|---|---|
| Scheduler | OneCycleLR (peak=0.0100, pct_start=0.3) |
| Epochs | 38 |
| Batch size | 128 |
| Label smoothing | 0.1 |
| Augmentation | RandomCrop(32,pad=4), RandomHorizontalFlip, ColorJitter |
| Gradient clipping | max_norm=1.0 |

## Results
| Test Accuracy    | 0.4025   |
|---|---|
| Macro Precision  | 0.4014  |
| Macro Recall     | 0.4025     |
| Macro F1         | 0.3935         |
| Test Loss        | 2.3768       |
| Parameters       | 1,184,180   |
| Memory (MB)      | 4.5173  |
| Inference (ms)   | 0.6992 |
| Training Time    | 533.0s (8.9 min) |

## Existing Baseline (for reference)
- Test Accuracy : 0.4233
- Macro F1      : 0.41195444400008774
- Parameters    : 311,796.0

## Pruning Compatibility
- `conv1` retains exactly 16 output channels — all pruning code unchanged.
- `forward_masked(x, mask)` interface preserved.
- `build_improved_pruned_model(model, retained_channels)` correctly transfers weights.
- Memory cost per channel updated to 2428 bytes (wider conv2 input slice: 64×3×3×4).

## Validation vs. Test Discipline
- Checkpoint selected using **validation** accuracy only.
- **Test subset was evaluated exactly once**, after checkpoint selection.
- Val and test indices are disjoint (verified programmatically).

## 70–80% Target Assessment
- Target reached: **NO**
- Achieved accuracy: 0.4025 (gap to 70% target: 0.2975)

## Why 70%+ is Hard Here
- The **16-channel bottleneck** (conv1) is the dominant constraint.
  With only 16 feature maps after the first conv, the network has limited
  representational power regardless of how wide the downstream layers are.
- CIFAR-100 has 100 classes, so even large modern nets struggle below 80%
  without residual connections, large ensembles, or heavy augmentation.
- Training on 10k images (1/5th of CIFAR-100) further limits generalisation.

## Recommendations for Next Experiment
1. **Residual connections** inside the 16-channel bottleneck: skip-connect
   the conv1 input to the conv2 input (requires 1×1 projection). This is
   compatible with the pruning contract because conv1 output channels are still 16.
2. **Full 50k training set**: ~5–8% accuracy gain expected from more data.
3. **Pre-trained backbone**: use a torchvision ResNet-18 up to the first
   residual group as the backbone and attach the 16-channel bottleneck
   as a plug-in layer for pruning.
4. **MixUp / CutMix** data augmentation: typically adds 2–3% on CIFAR-100.
5. **Longer training** (100 epochs) with CosineAnnealingLR + warm-up.