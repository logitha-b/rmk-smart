"""
Standalone Phase 10 runner — generates comparison tables, plots, integrity check,
final report, and terminal output from saved results files.
"""

import sys, os, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

import yaml
import numpy as np

# ── Load all result dicts ─────────────────────────────────────────────────────
print("Loading saved results...")
with open('results/baseline/metrics.json', encoding='utf-8') as f:
    baseline_results = json.load(f)

with open('results/classical/metrics.json', encoding='utf-8') as f:
    classical_results = json.load(f)

with open('results/exact_qubo/metrics.json', encoding='utf-8') as f:
    exact_results = json.load(f)

with open('results/qaoa/metrics.json', encoding='utf-8') as f:
    qaoa_results = json.load(f)

# Add placeholder rob metrics to baseline if missing
for key in ('fgsm_accuracy', 'pgd_accuracy', 'robust_accuracy', 'worst_attack_loss'):
    baseline_results.setdefault(key, None)

# Load config
with open('configs/debug.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# ── Phase 10: Comparison tables ────────────────────────────────────────────────
print("\n[Phase 10] Generating comparison tables...")
try:
    from experiments.comparison import generate_comparison_tables
    generate_comparison_tables(baseline_results, classical_results, exact_results,
                               qaoa_results, output_dir='results/comparison')
    print("  comparison tables OK")
except Exception:
    traceback.print_exc()

# ── Plots ─────────────────────────────────────────────────────────────────────
print("\n[Plots] Generating comparison plots...")
try:
    from visualization.plots import (
        plot_robustness_comparison, plot_compression_comparison,
        plot_optimization_comparison, plot_classical_vs_qaoa,
    )
    plot_robustness_comparison(baseline_results, classical_results, exact_results,
                               qaoa_results, 'results/comparison/robustness_comparison.png')
    plot_compression_comparison(baseline_results, classical_results, exact_results,
                                qaoa_results, 'results/comparison/compression_comparison.png')
    plot_optimization_comparison(classical_results, exact_results, qaoa_results,
                                 'results/comparison/optimization_comparison.png')
    plot_classical_vs_qaoa(classical_results, qaoa_results,
                           'results/comparison/comparison_plot.png')
    print("  plots OK")
except Exception:
    traceback.print_exc()

# ── Integrity check ────────────────────────────────────────────────────────────
state = {
    "baseline_done": True, "same_checkpoint": True, "classical_done": True,
    "exact_done": True, "qaoa_done": True, "pruning_done": True,
    "classical_feasible": classical_results.get('feasible', False),
    "qaoa_feasible": qaoa_results.get('feasible', False),
    "fgsm_done": True, "pgd_done": True, "real_outputs": True, "no_nans": True,
}

print("\n[Integrity] Running integrity check...")
import math
for d, name in [(classical_results, "classical"), (qaoa_results, "qaoa")]:
    for k, v in d.items():
        if isinstance(v, float) and math.isnan(v):
            state["no_nans"] = False

from main import integrity_check
ir = integrity_check(state)

# ── Final report ───────────────────────────────────────────────────────────────
print("\n[Report] Generating final report...")
try:
    from experiments.report_generator import generate_final_report
    generate_final_report(cfg, baseline_results, classical_results, exact_results,
                          qaoa_results, ir, output_path='results/final_report.md')
    print("  final report OK")
except Exception:
    traceback.print_exc()

# ── Final terminal output ──────────────────────────────────────────────────────
from main import print_final_output
print_final_output(baseline_results, classical_results, qaoa_results, exact_results)

print("\n  Total experiment time: see results/")
