"""
Baseline model training and evaluation experiment.

Responsibilities:
  1. Set random seeds.
  2. Train SmallCNN on CIFAR-100 (or load existing checkpoint).
  3. Evaluate the trained model on the full test set.
  4. Verify that training succeeded (model actually learned).
  5. Save baseline metrics.
  6. Return the trained model and metrics for downstream experiments.
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, Any, Tuple

from models.cnn       import build_model
from models.model_utils import (
    count_parameters,
    model_memory_mb,
    measure_inference_time,
    load_checkpoint,
    verify_model_shapes,
)
from training.train    import train_model, get_cifar100_loaders
from training.evaluate import evaluate_full


def set_seeds(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"  [seed] All seeds set to {seed}")


def run_baseline(cfg: Dict[str, Any]) -> Tuple[torch.nn.Module, Dict[str, Any], torch.utils.data.DataLoader]:
    """
    Main baseline experiment.

    Returns:
        model        : trained SmallCNN (best checkpoint loaded)
        metrics      : dict of all baseline metrics
        test_loader  : DataLoader for the test set (reused by all experiments)
    """
    seed      = cfg.get("seed",           42)
    save_path = cfg.get("checkpoint_path", "results/checkpoints/baseline_model.pt")
    force_retrain = cfg.get("force_retrain", False)
    device_str    = "cuda" if torch.cuda.is_available() else "cpu"
    device        = torch.device(device_str)

    set_seeds(seed)
    print(f"\n{'='*60}")
    print("  PHASE 2: BASELINE MODEL")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # ── Build data loaders ────────────────────────────────────────────────────
    _, test_loader = get_cifar100_loaders(cfg)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes=100).to(device)

    # ── Train or load ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path) and not force_retrain:
        print(f"  [baseline] Loading existing checkpoint: {save_path}")
        meta = load_checkpoint(model, save_path, device)
        history = meta if meta else {}
        print(f"  [baseline] Checkpoint metadata: {meta}")
    else:
        print("  [baseline] Training from scratch ...")
        history = train_model(model, cfg, device, save_path)
        # Reload best checkpoint
        load_checkpoint(model, save_path, device)

    # ── Shape verification ────────────────────────────────────────────────────
    ok = verify_model_shapes(model, device)
    if not ok:
        raise RuntimeError("Baseline model shape verification FAILED.")

    # ── Full evaluation ───────────────────────────────────────────────────────
    model.eval()
    metrics = evaluate_full(model, test_loader, device)

    # ── Training success check ────────────────────────────────────────────────
    # Random chance on CIFAR-100 = 1%.
    # Debug mode: allow lower threshold (limited data + few epochs).
    # Research mode: require > 5%.
    debug_mode = cfg.get("debug_mode", False)
    MINIMUM_ACCURACY_THRESHOLD = 0.008 if debug_mode else 0.05
    if metrics["accuracy"] < MINIMUM_ACCURACY_THRESHOLD:
        raise RuntimeError(
            f"Baseline training FAILED. "
            f"Test accuracy {metrics['accuracy']:.4f} is below threshold "
            f"{MINIMUM_ACCURACY_THRESHOLD} ({'debug' if debug_mode else 'research'} mode). "
            "Aborting experiment to prevent misleading pruning results."
        )
    if debug_mode and metrics["accuracy"] < 0.05:
        print(f"  [baseline] DEBUG WARNING: accuracy {metrics['accuracy']:.4f} is low "
              f"(expected in debug mode with few epochs/samples). "
              f"Do NOT use debug results in the final research report.")

    # ── Print baseline summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  BASELINE MODEL")
    print(f"  {'─'*40}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Precision      : {metrics['precision']:.4f}")
    print(f"  Recall         : {metrics['recall']:.4f}")
    print(f"  F1             : {metrics['f1']:.4f}")
    print(f"  Loss           : {metrics['loss']:.4f}")
    print(f"  Parameters     : {metrics['parameters']:,}")
    print(f"  Memory         : {metrics['memory_mb']:.4f} MB")
    print(f"  Inference Time : {metrics['inference_ms']:.4f} ms")
    print(f"{'='*60}")

    # ── Save baseline metrics ──────────────────────────────────────────────────
    os.makedirs("results/baseline", exist_ok=True)
    metrics_out = {k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v)
                   for k, v in metrics.items()}
    metrics_out["training_history"] = {
        k: [float(x) for x in v] if isinstance(v, list) else v
        for k, v in history.items()
    }
    metrics_out["device"]   = device_str
    metrics_out["seed"]     = seed

    with open("results/baseline/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # CSV summary
    import pandas as pd
    row = {
        "accuracy":   metrics["accuracy"],
        "precision":  metrics["precision"],
        "recall":     metrics["recall"],
        "f1":         metrics["f1"],
        "loss":       metrics["loss"],
        "parameters": metrics["parameters"],
        "memory_mb":  metrics["memory_mb"],
        "inference_ms": metrics["inference_ms"],
    }
    pd.DataFrame([row]).to_csv("results/baseline/metrics.csv", index=False)

    print("  [baseline] Metrics saved to results/baseline/")

    return model, metrics, test_loader
