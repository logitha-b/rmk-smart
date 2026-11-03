"""
Exact exhaustive QUBO solver.

With 16 binary variables there are 2^16 = 65,536 possible bitstrings.
This solver enumerates ALL feasible candidates and selects the one
with the highest QUBO objective value.

This is NOT the conventional pruning baseline.
It is the ground-truth optimizer for the QUBO formulation.

Performance:
    With numpy vectorization, evaluating 65,536 bitstrings takes < 1 second.
"""

import time
import numpy as np
from typing import Dict, Any, List, Tuple

from optimization.qubo import evaluate_qubo_vectorized
from pruning.pruning_utils import check_feasibility, mask_to_channel_list


def enumerate_all_bitstrings(n: int) -> np.ndarray:
    """
    Generate all 2^n binary vectors of length n.

    Returns:
        Array of shape (2^n, n), dtype uint8.
    """
    N = 2 ** n
    bits = np.zeros((N, n), dtype=np.uint8)
    for k in range(N):
        for b in range(n):
            bits[k, b] = (k >> b) & 1
    return bits


def exact_qubo_solve(
    lin:  np.ndarray,
    quad: np.ndarray,
    w:    np.ndarray,
    M:    float,
    num_channels: int = 16,
) -> Dict[str, Any]:
    """
    Exhaustive exact QUBO solver.

    Enumerates all 2^num_channels bitstrings, filters for feasibility,
    and returns the feasible bitstring with the highest objective.

    Args:
        lin          : linear QUBO coefficients, shape (n,)
        quad         : quadratic QUBO coefficients, shape (n, n)
        w            : memory cost per channel, shape (n,)
        M            : memory budget
        num_channels : n = 16

    Returns:
        Dict with keys:
            best_mask       : np.ndarray, shape (n,)
            best_objective  : float
            retained_channels : list[int]
            pruned_channels   : list[int]
            runtime_s         : float
            n_feasible        : int
            n_evaluated       : int
    """
    n       = num_channels
    t_start = time.time()

    # ── Generate all bitstrings ───────────────────────────────────────────────
    N    = 2 ** n
    bits = enumerate_all_bitstrings(n).astype(np.float64)

    # ── Vectorized QUBO evaluation ────────────────────────────────────────────
    # Obj_k = bits[k] · lin  +  0.5 * bits[k]^T · quad · bits[k]
    linear_vals = bits @ lin                                   # (N,)
    quad_vals   = 0.5 * np.einsum('ni,ij,nj->n', bits, quad, bits)  # (N,)
    obj_vals    = linear_vals + quad_vals                      # (N,)

    # ── Feasibility filter ────────────────────────────────────────────────────
    mem_usage = bits @ w                       # (N,)
    feasible  = mem_usage <= M                 # boolean mask, shape (N,)
    n_feasible = int(feasible.sum())

    if n_feasible == 0:
        runtime_s = time.time() - t_start
        print("  [exact] WARNING: No feasible solution found! Relaxing constraint.")
        # Fall back: return the mask that minimizes memory violation
        violation  = np.maximum(mem_usage - M, 0.0)
        best_k     = np.argmin(violation)
        best_mask  = bits[best_k].astype(np.float64)
        best_obj   = float(obj_vals[best_k])
    else:
        # Among feasible, pick the one with highest objective
        feasible_objs = np.where(feasible, obj_vals, -np.inf)
        best_k        = int(np.argmax(feasible_objs))
        best_mask     = bits[best_k].astype(np.float64)
        best_obj      = float(obj_vals[best_k])

    runtime_s = time.time() - t_start

    retained = mask_to_channel_list(best_mask)
    pruned   = [i for i in range(n) if i not in retained]

    print(f"  [exact] Evaluated {N} bitstrings  |  Feasible: {n_feasible}")
    print(f"  [exact] Best mask     : {best_mask.astype(int).tolist()}")
    print(f"  [exact] Best objective: {best_obj:.6f}")
    print(f"  [exact] Retained      : {retained}")
    print(f"  [exact] Runtime       : {runtime_s:.4f}s")

    return {
        "method":             "exact_qubo",
        "best_mask":          best_mask,
        "best_objective":     best_obj,
        "retained_channels":  retained,
        "pruned_channels":    pruned,
        "runtime_s":          runtime_s,
        "n_feasible":         n_feasible,
        "n_evaluated":        N,
        "feasible":           n_feasible > 0,
    }
