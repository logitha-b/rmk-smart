"""
generate_reports.py
===================
Re-generates comparison_report.md, comparison_report.csv, and
diagnostic_report.md from already-saved metrics.json and subset_indices.json.

Runs the final validation checklist against the saved checkpoint.

Usage:
    python generate_reports.py
"""

import os, sys, json, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_small_baseline import (
    generate_comparison,
    generate_diagnostic_report,
    run_final_validation,
)
from experiments.baseline_small import ImprovedSmallCNN

OUT_DIR   = "results/baseline_small"
CKPT_PATH = os.path.join(OUT_DIR, "checkpoints", "baseline_model.pt")
CFG_PATH  = os.path.join(OUT_DIR, "dataset_config.json")

def main():
    # Load metrics
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    # Remove nested training_history for flat metrics dict
    metrics_flat = {k: v for k, v in metrics.items() if k != "training_history"}

    # Load subset indices
    with open(os.path.join(OUT_DIR, "subset_indices.json")) as f:
        idx = json.load(f)
    train_idx = idx["train_indices"]
    val_idx   = idx["val_indices"]
    test_idx  = idx["test_indices"]

    # Rebuild config for diagnostic report
    cfg = {
        "seed":            metrics_flat.get("seed", 42),
        "epochs":          int(metrics["training_history"]["best_epoch"]),
        "batch_size":      128,
        "learning_rate":   1e-3,
        "weight_decay":    0.01,
        "label_smoothing": 0.1,
        "small_train_size": metrics_flat["n_train"],
        "small_val_size":   metrics_flat["n_val"],
        "small_test_size":  metrics_flat["n_test"],
    }

    # Generate comparison
    print("[1/3] Generating comparison report …")
    md = generate_comparison(metrics_flat, OUT_DIR)
    print(md)

    # Generate diagnostic
    print("\n[2/3] Generating diagnostic report …")
    generate_diagnostic_report(cfg, metrics_flat, OUT_DIR)
    print("      Done.")

    # Final validation checklist
    print("\n[3/3] Running final validation checklist …")
    device = torch.device("cpu")
    model  = ImprovedSmallCNN(num_classes=100).to(device)
    payload = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    run_final_validation(model, train_idx, val_idx, test_idx, metrics_flat)

    print(f"\nAll report files written to {OUT_DIR}/")
    print("Files:")
    for fname in [
        "metrics.json", "metrics.csv", "training_history.csv",
        "training_curve.png", "accuracy_curve.png",
        "dataset_config.json", "subset_indices.json",
        "diagnostic_report.md", "comparison_report.md", "comparison_report.csv",
        "checkpoints/baseline_model.pt",
    ]:
        fpath = os.path.join(OUT_DIR, fname)
        mark  = "OK" if os.path.exists(fpath) else "MISSING"
        print(f"  [{mark}] {fname}")

if __name__ == "__main__":
    main()
