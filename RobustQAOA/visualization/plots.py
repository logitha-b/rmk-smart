"""
Visualization module — all experiment plots.

Generates:
  1. Training curve
  2. Channel importance bar chart
  3. Pairwise synergy heatmap
  4. Robustness comparison bar chart
  5. Compression comparison bar chart
  6. Optimization comparison (objective + runtime)
  7. Defender-attacker iteration curves
  8. Classical vs QAOA metric comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import Dict, Any, List, Optional


# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
CLASSICAL_COLOR = "#2196F3"   # blue
QAOA_COLOR      = "#FF5722"   # orange
EXACT_COLOR     = "#4CAF50"   # green
BASELINE_COLOR  = "#9E9E9E"   # gray


def save_fig(path: str, dpi: int = 150) -> None:
    """Save current figure and close."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [plot] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Training curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(history: Dict[str, Any], output_path: str) -> None:
    """Plot training and validation loss/accuracy curves."""
    if not history.get("train_loss"):
        print("  [plot] No training history available for curve.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#1976D2")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#1976D2")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="#F44336")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()

    plt.suptitle("Baseline Model Training Curves")
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Channel importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_channel_importance(importance: np.ndarray, output_path: str) -> None:
    """Bar chart of channel importance scores."""
    n = len(importance)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [CLASSICAL_COLOR if v >= 0 else "#F44336" for v in importance]
    bars = ax.bar(range(n), importance, color=colors)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Importance (c_i = f(full) - f(-i))")
    ax.set_title("Channel Importance Scores (conv1, 16 channels)")
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(n))
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synergy heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_synergy_heatmap(synergy: np.ndarray, output_path: str) -> None:
    """Heatmap of pairwise channel synergy matrix."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        synergy, ax=ax,
        cmap="RdBu_r", center=0,
        xticklabels=range(synergy.shape[1]),
        yticklabels=range(synergy.shape[0]),
        annot=False, fmt=".3f",
    )
    ax.set_title("Pairwise Channel Synergy Matrix (Q_ij)")
    ax.set_xlabel("Channel j")
    ax.set_ylabel("Channel i")
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Robustness comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_robustness_comparison(
    baseline:  Dict[str, Any],
    classical: Dict[str, Any],
    exact:     Dict[str, Any],
    qaoa:      Dict[str, Any],
    output_path: str,
) -> None:
    """Grouped bar chart: clean / FGSM / PGD / robust accuracy per model."""
    models  = ["Original", "Classical", "Exact QUBO", "QAOA"]
    results = [baseline, classical, exact, qaoa]

    metrics = ["accuracy", "fgsm_accuracy", "pgd_accuracy", "robust_accuracy"]
    labels  = ["Clean", "FGSM", "PGD", "Robust"]
    colors  = [BASELINE_COLOR, CLASSICAL_COLOR, EXACT_COLOR, QAOA_COLOR]

    x   = np.arange(len(models))
    w   = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))

    for j, (m, lbl, c) in enumerate(zip(metrics, labels, colors)):
        vals = [r.get(m, float("nan")) for r in results]
        ax.bar(x + j * w - 1.5 * w, vals, w, label=lbl, color=c, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy")
    ax.set_title("Adversarial Robustness Comparison")
    ax.legend()
    ax.set_ylim(0, 1)
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Compression comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_compression_comparison(
    baseline:  Dict[str, Any],
    classical: Dict[str, Any],
    exact:     Dict[str, Any],
    qaoa:      Dict[str, Any],
    output_path: str,
) -> None:
    """Bar chart comparing parameter count and memory across models."""
    models  = ["Original", "Classical", "Exact QUBO", "QAOA"]
    results = [baseline, classical, exact, qaoa]
    colors  = [BASELINE_COLOR, CLASSICAL_COLOR, EXACT_COLOR, QAOA_COLOR]

    params = [r.get("parameters", 0) for r in results]
    mems   = [r.get("memory_mb",  0) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(models, params, color=colors, alpha=0.85)
    axes[0].set_title("Parameter Count")
    axes[0].set_ylabel("Parameters")
    for i, v in enumerate(params):
        axes[0].text(i, v * 1.01, f"{v:,}", ha="center", fontsize=8)

    axes[1].bar(models, mems, color=colors, alpha=0.85)
    axes[1].set_title("Memory Usage (MB)")
    axes[1].set_ylabel("Memory (MB)")
    for i, v in enumerate(mems):
        axes[1].text(i, v * 1.01, f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle("Model Compression Comparison")
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Optimization comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimization_comparison(
    classical: Dict[str, Any],
    exact:     Dict[str, Any],
    qaoa:      Dict[str, Any],
    output_path: str,
) -> None:
    """Bar chart: objective value and runtime per optimization method."""
    methods = ["Classical", "Exact QUBO", "QAOA"]
    results = [classical, exact, qaoa]
    colors  = [CLASSICAL_COLOR, EXACT_COLOR, QAOA_COLOR]

    objs     = [r.get("qubo_objective", 0)  for r in results]
    runtimes = [r.get("runtime_s",      0)  for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(methods, objs, color=colors, alpha=0.85)
    axes[0].set_title("QUBO Objective (maximize)")
    axes[0].set_ylabel("Objective Value")

    axes[1].bar(methods, runtimes, color=colors, alpha=0.85)
    axes[1].set_title("Optimization Runtime (s)")
    axes[1].set_ylabel("Time (seconds)")
    axes[1].set_yscale("log")

    plt.suptitle("Optimization Method Comparison")
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Iteration curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_iteration_curves(df: pd.DataFrame, output_path: str) -> None:
    """Line plots showing metric evolution across defender-attacker rounds."""
    if df.empty:
        print("  [plot] Empty iteration dataframe, skipping.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    rounds = df["round"].values

    for ax, c_col, q_col, title in [
        (axes[0, 0], "classical_objective",      "qaoa_objective",       "QUBO Objective"),
        (axes[0, 1], "classical_clean_accuracy",  "qaoa_clean_accuracy",  "Clean Accuracy"),
        (axes[1, 0], "classical_robust_accuracy", "qaoa_robust_accuracy", "Robust Accuracy"),
        (axes[1, 1], "classical_n_retained",      "qaoa_n_retained",      "Retained Channels"),
    ]:
        if c_col in df.columns:
            ax.plot(rounds, df[c_col], marker="o", color=CLASSICAL_COLOR, label="Classical")
        if q_col in df.columns:
            ax.plot(rounds, df[q_col], marker="s", color=QAOA_COLOR,      label="QAOA")
        ax.set_xlabel("Round")
        ax.set_title(title)
        ax.legend()

    plt.suptitle("Defender-Attacker Loop: Evolution Across Rounds")
    save_fig(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Classical vs QAOA metric comparison (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────

def plot_classical_vs_qaoa(
    classical: Dict[str, Any],
    qaoa:      Dict[str, Any],
    output_path: str,
) -> None:
    """Horizontal grouped bar chart for key metrics."""
    metric_keys   = ["accuracy", "f1", "fgsm_accuracy", "pgd_accuracy", "robust_accuracy"]
    metric_labels = ["Clean Acc", "Macro F1", "FGSM Acc", "PGD Acc", "Robust Acc"]

    c_vals = [classical.get(k, float("nan")) for k in metric_keys]
    q_vals = [qaoa.get(k,     float("nan")) for k in metric_keys]

    y   = np.arange(len(metric_labels))
    h   = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(y - h / 2, c_vals, h, label="Classical", color=CLASSICAL_COLOR, alpha=0.85)
    ax.barh(y + h / 2, q_vals, h, label="QAOA",      color=QAOA_COLOR,      alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Value")
    ax.set_title("Classical vs QAOA: Key Metrics")
    ax.legend()
    ax.set_xlim(0, 1)
    save_fig(output_path)
