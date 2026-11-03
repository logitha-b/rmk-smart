"""
Comparison table generation.

Generates:
  1. Classical vs QAOA comparison table
  2. QAOA vs Exact QUBO table
  3. Adversarial robustness table
  4. Memory/compression table
  5. Optimization comparison table

All values come directly from the experiment result dicts.
No values are fabricated or hard-coded.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def generate_comparison_tables(
    baseline_results:   Dict[str, Any],
    classical_results:  Dict[str, Any],
    exact_results:      Dict[str, Any],
    qaoa_results:       Dict[str, Any],
    output_dir:         str = "results/comparison",
) -> None:
    """
    Generate and save all comparison tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Classical vs QAOA main table ──────────────────────────────────────
    main_table = _build_main_table(classical_results, qaoa_results)

    # ── 2. QAOA vs Exact QUBO ────────────────────────────────────────────────
    qubo_table = _build_qubo_table(exact_results, qaoa_results)

    # ── 3. Adversarial robustness table ──────────────────────────────────────
    rob_table  = _build_robustness_table(
        baseline_results, classical_results, exact_results, qaoa_results
    )

    # ── 4. Compression table ─────────────────────────────────────────────────
    comp_table = _build_compression_table(
        baseline_results, classical_results, exact_results, qaoa_results
    )

    # ── 5. Optimization table ────────────────────────────────────────────────
    opt_table  = _build_optimization_table(classical_results, exact_results, qaoa_results)

    # ── Save all ──────────────────────────────────────────────────────────────
    main_table.to_csv(  os.path.join(output_dir, "comparison.csv"),   index=False)
    qubo_table.to_csv(  os.path.join(output_dir, "qubo_comparison.csv"), index=False)
    rob_table.to_csv(   os.path.join(output_dir, "robustness_comparison.csv"), index=False)
    comp_table.to_csv(  os.path.join(output_dir, "compression_comparison.csv"), index=False)
    opt_table.to_csv(   os.path.join(output_dir, "optimization_comparison.csv"), index=False)

    # ── Generate markdown report ──────────────────────────────────────────────
    md = _generate_comparison_md(main_table, qubo_table, rob_table, comp_table, opt_table,
                                  classical_results, qaoa_results, exact_results)
    with open(os.path.join(output_dir, "comparison.md"), "w") as f:
        f.write(md)

    # ── Save combined JSON ────────────────────────────────────────────────────
    combined = {
        "classical": _serialize(classical_results),
        "exact_qubo": _serialize(exact_results),
        "qaoa":       _serialize(qaoa_results),
    }
    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # ── Print main table to terminal ──────────────────────────────────────────
    _print_main_table(main_table)
    _print_qubo_table(qubo_table)


def _g(d: Dict, key: str, default=float('nan')):
    """Safe getter that returns nan for missing keys."""
    v = d.get(key, default)
    if v is None:
        return float('nan')
    return v


def _build_main_table(classical: Dict, qaoa: Dict) -> pd.DataFrame:
    """Build the CLASSICAL vs QAOA comparison table."""
    metrics = [
        ("Clean Accuracy",       "accuracy"),
        ("Macro Precision",      "precision"),
        ("Macro Recall",         "recall"),
        ("Macro F1",             "f1"),
        ("Test Loss",            "loss"),
        ("Parameters",           "parameters"),
        ("Parameter Reduction %","param_reduction_pct"),
        ("Memory (MB)",          "memory_mb"),
        ("Memory Reduction %",   "mem_reduction_pct"),
        ("Inference Time (ms)",  "inference_ms"),
        ("FGSM Accuracy",        "fgsm_accuracy"),
        ("PGD Accuracy",         "pgd_accuracy"),
        ("Robust Accuracy",      "robust_accuracy"),
        ("Worst Attack Loss",    "worst_attack_loss"),
        ("Retained Channels",    "retained_channels"),
        ("Pruned Channels",      "pruned_channels"),
        ("Optimization Runtime (s)", "runtime_s"),
        ("QUBO Objective",       "qubo_objective"),
    ]

    rows = []
    for label, key in metrics:
        c_val = _g(classical, key)
        q_val = _g(qaoa,      key)

        # For lists (channels), compute count difference
        if isinstance(c_val, list):
            c_disp = len(c_val)
            q_disp = len(q_val) if isinstance(q_val, list) else float('nan')
            diff   = q_disp - c_disp if not np.isnan(q_disp) else float('nan')
        else:
            c_disp = c_val
            q_disp = q_val
            try:
                diff = float(q_val) - float(c_val)
            except (TypeError, ValueError):
                diff = float('nan')

        rows.append({
            "Metric":                     label,
            "Classical / Actual Output":  _fmt(c_disp),
            "QAOA Output":                _fmt(q_disp),
            "Difference (QAOA - Classical)": _fmt(diff),
        })

    return pd.DataFrame(rows)


def _build_qubo_table(exact: Dict, qaoa: Dict) -> pd.DataFrame:
    """Build QAOA vs Exact QUBO table."""
    exact_obj = _g(exact, "qubo_objective")
    qaoa_obj  = _g(qaoa,  "qubo_objective")
    gap       = exact_obj - qaoa_obj if not (np.isnan(exact_obj) or np.isnan(qaoa_obj)) else float('nan')

    exact_ret = exact.get("retained_channels", [])
    qaoa_ret  = qaoa.get("retained_channels",  [])

    rows = [
        {"Metric": "Objective",        "Exact QUBO": _fmt(exact_obj),       "QAOA": _fmt(qaoa_obj),       "Difference": _fmt(qaoa_obj - exact_obj if not np.isnan(gap) else float('nan'))},
        {"Metric": "Retained Channels","Exact QUBO": len(exact_ret),         "QAOA": len(qaoa_ret),         "Difference": len(qaoa_ret) - len(exact_ret)},
        {"Metric": "Pruned Channels",  "Exact QUBO": 16 - len(exact_ret),   "QAOA": 16 - len(qaoa_ret),   "Difference": (16 - len(qaoa_ret)) - (16 - len(exact_ret))},
        {"Metric": "Runtime (s)",      "Exact QUBO": _fmt(_g(exact, "runtime_s")), "QAOA": _fmt(_g(qaoa, "runtime_s")), "Difference": _fmt(_g(qaoa, "runtime_s") - _g(exact, "runtime_s"))},
    ]

    # Exact optimum found by QAOA?
    exact_mask = exact.get("best_mask", None)
    qaoa_mask  = qaoa.get("mask",       None)
    if exact_mask is not None and qaoa_mask is not None:
        em = np.array(exact_mask).astype(int).tolist() if not isinstance(exact_mask, list) else exact_mask
        qm = np.array(qaoa_mask).astype(int).tolist()  if not isinstance(qaoa_mask,  list) else qaoa_mask
        found_exact = "YES" if em == qm else "NO"
    else:
        found_exact = "UNKNOWN"

    rows.append({"Metric": "Exact Optimum Found", "Exact QUBO": "YES", "QAOA": found_exact, "Difference": "—"})

    df = pd.DataFrame(rows)
    return df


def _build_robustness_table(base, classical, exact, qaoa) -> pd.DataFrame:
    rows = []
    for label, d in [("Original", base), ("Classical", classical),
                     ("Exact QUBO", exact), ("QAOA", qaoa)]:
        rows.append({
            "Model":            label,
            "Clean Accuracy":   _fmt(_g(d, "accuracy")),
            "FGSM Accuracy":    _fmt(_g(d, "fgsm_accuracy")),
            "PGD Accuracy":     _fmt(_g(d, "pgd_accuracy")),
            "Robust Accuracy":  _fmt(_g(d, "robust_accuracy")),
        })
    return pd.DataFrame(rows)


def _build_compression_table(base, classical, exact, qaoa) -> pd.DataFrame:
    rows = []
    for label, d in [("Original", base), ("Classical", classical),
                     ("Exact QUBO", exact), ("QAOA", qaoa)]:
        rows.append({
            "Model":                  label,
            "Parameters":             _g(d, "parameters"),
            "Parameter Reduction %":  _fmt(_g(d, "param_reduction_pct", 0.0)),
            "Memory (MB)":            _fmt(_g(d, "memory_mb")),
            "Memory Reduction %":     _fmt(_g(d, "mem_reduction_pct", 0.0)),
        })
    return pd.DataFrame(rows)


def _build_optimization_table(classical, exact, qaoa) -> pd.DataFrame:
    rows = []
    for label, d in [("Classical", classical), ("Exact QUBO", exact), ("QAOA", qaoa)]:
        ret = d.get("retained_channels", [])
        rows.append({
            "Method":             label,
            "Objective":          _fmt(_g(d, "qubo_objective")),
            "Runtime (s)":        _fmt(_g(d, "runtime_s")),
            "Feasible":           str(d.get("feasible", "N/A")),
            "Retained Channels":  len(ret) if isinstance(ret, list) else ret,
        })
    return pd.DataFrame(rows)


def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):  return "N/A"
        return f"{v:.4f}"
    if isinstance(v, list):
        return str(v)
    return str(v)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table without requiring tabulate."""
    cols = list(df.columns)
    col_widths = {c: max(len(str(c)), df[c].astype(str).str.len().max()) for c in cols}

    header = "| " + " | ".join(str(c).ljust(col_widths[c]) for c in cols) + " |"
    sep    = "|-" + "-|-".join("-" * col_widths[c] for c in cols) + "-|"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]).ljust(col_widths[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def _generate_comparison_md(main, qubo, rob, comp, opt, classical, qaoa, exact) -> str:
    lines = [
        "# CLASSICAL vs QAOA EXPERIMENTAL COMPARISON",
        "",
        "## Main Comparison Table",
        "",
        _df_to_markdown(main),
        "",
        "## QAOA vs Exact QUBO",
        "",
        _df_to_markdown(qubo),
        "",
        "## Adversarial Robustness Comparison",
        "",
        _df_to_markdown(rob),
        "",
        "## Compression Comparison",
        "",
        _df_to_markdown(comp),
        "",
        "## Optimization Comparison",
        "",
        _df_to_markdown(opt),
        "",
        "---",
        f"*Retained channels — Classical: {classical.get('retained_channels')}*",
        f"*Retained channels — QAOA: {qaoa.get('retained_channels')}*",
        f"*Retained channels — Exact QUBO: {exact.get('retained_channels')}*",
    ]
    return "\n".join(lines)


def _print_main_table(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  CLASSICAL vs QAOA EXPERIMENTAL COMPARISON")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}")


def _print_qubo_table(df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print("  QAOA vs EXACT QUBO")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"{'='*60}")


def _serialize(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):               out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)): out[k] = float(v)
        else:                                        out[k] = v
    return out
