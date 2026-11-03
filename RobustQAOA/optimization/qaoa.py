"""
QAOA (Quantum Approximate Optimization Algorithm) implementation.

Uses Qiskit / Qiskit Aer to:
  1. Construct the cost Hamiltonian from the QUBO coefficients.
  2. Build the QAOA ansatz circuit of depth p.
  3. Optimize the variational parameters using a classical optimizer (COBYLA).
  4. Sample bitstrings from the optimized circuit.
  5. Decode bitstrings, check feasibility, select the best feasible candidate.

Sign convention:
  - The QAOA cost Hamiltonian H_C is built to be MINIMIZED.
  - Since our QUBO is a MAXIMIZATION problem, we pass the NEGATED objective
    to the cost Hamiltonian construction.
  - All reported objective values are converted back to MAXIMIZATION form.

Circuit structure for depth p:
  |ψ₀⟩ = H^⊗n |0...0⟩   (uniform superposition)
  For layer k = 1 ... p:
      Apply cost unitary: U_C(γ_k) = e^{-i γ_k H_C}
      Apply mixer unitary: U_M(β_k) = e^{-i β_k H_M}  where H_M = Σ X_i
  Measure in computational basis.

Each phase gate e^{-i γ H_C} is decomposed into:
  - RZZ(2 γ J_ij) gates for quadratic terms
  - RZ(2 γ h_i) gates for linear terms

The mixer is: ⊗_i RX(2β_i) gates.
"""

import time
import warnings
import numpy as np
from typing import Dict, Any, List, Tuple

# ── Qiskit imports ────────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_aer import AerSimulator
    from qiskit.compiler import transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn(
        "Qiskit or Qiskit Aer not found. QAOA will fall back to classical simulation.",
        RuntimeWarning,
    )

try:
    from scipy.optimize import minimize as scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from optimization.qubo import qubo_to_ising, evaluate_qubo_vectorized
from pruning.pruning_utils import check_feasibility, mask_to_channel_list


# ─────────────────────────────────────────────────────────────────────────────
# Circuit construction
# ─────────────────────────────────────────────────────────────────────────────

def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    p: int,
    gammas: np.ndarray,
    betas:  np.ndarray,
) -> "QuantumCircuit":
    """
    Build a QAOA circuit for the Ising Hamiltonian H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j.

    Args:
        h      : Ising field coefficients, shape (n,)
        J      : Ising coupling coefficients, shape (n, n), symmetric, upper-tri used
        p      : QAOA depth (number of layers)
        gammas : cost layer parameters, shape (p,)
        betas  : mixer layer parameters, shape (p,)

    Returns:
        QuantumCircuit with measurement gates.
    """
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit not available. Cannot build QAOA circuit.")

    n   = len(h)
    qc  = QuantumCircuit(n, n)

    # Initial state: uniform superposition
    qc.h(range(n))

    for layer in range(p):
        gamma_k = float(gammas[layer])
        beta_k  = float(betas[layer])

        # ── Cost unitary U_C(γ_k) ────────────────────────────────────────────
        # Linear terms: RZ(2 γ h_i) on qubit i
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2.0 * gamma_k * h[i], i)

        # Quadratic terms: RZZ(2 γ J_ij) on qubits (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    # RZZ(θ) = CNOT · RZ(θ) · CNOT
                    qc.cx(i, j)
                    qc.rz(2.0 * gamma_k * J[i, j], j)
                    qc.cx(i, j)

        # ── Mixer unitary U_M(β_k) ────────────────────────────────────────────
        for i in range(n):
            qc.rx(2.0 * beta_k, i)

    # Measurement
    qc.measure(range(n), range(n))

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# QAOA solver
# ─────────────────────────────────────────────────────────────────────────────

def run_qaoa(
    lin:  np.ndarray,
    quad: np.ndarray,
    w:    np.ndarray,
    M:    float,
    cfg:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full QAOA pipeline for QUBO channel selection.

    Args:
        lin  : linear QUBO coefficients (maximization)
        quad : quadratic QUBO coefficients (maximization)
        w    : memory cost per channel
        M    : memory budget
        cfg  : configuration dict

    Returns:
        Results dict with best feasible bitstring and metrics.
    """
    if not QISKIT_AVAILABLE:
        return _qaoa_fallback(lin, quad, w, M, cfg)

    p            = cfg.get("qaoa_depth",          1)
    shots        = cfg.get("qaoa_shots",           512)
    opt_name     = cfg.get("qaoa_optimizer",       "COBYLA")
    max_iter     = cfg.get("qaoa_max_iterations",  100)
    seed         = cfg.get("seed",                 42)
    num_channels = len(lin)

    t_start = time.time()

    # ── Convert QUBO (maximize) → Ising (minimize) ────────────────────────────
    h, J, offset = qubo_to_ising(lin, quad)

    # ── Simulator ────────────────────────────────────────────────────────────
    simulator = AerSimulator(method='statevector')

    # ── Classical optimizer objective ────────────────────────────────────────
    n_params    = 2 * p
    iter_count  = [0]
    best_params = [None]
    best_energy = [np.inf]

    def objective(params):
        gammas = params[:p]
        betas  = params[p:]

        qc  = build_qaoa_circuit(h, J, p, gammas, betas)
        tqc = transpile(qc, simulator, optimization_level=0)

        job    = simulator.run(tqc, shots=shots, seed_simulator=seed)
        counts = job.result().get_counts()

        # Expected energy: sum over sampled bitstrings
        energy = 0.0
        total  = sum(counts.values())
        for bitstring, count in counts.items():
            # Qiskit bitstring is big-endian (qubit n-1 leftmost)
            bits = np.array([int(b) for b in reversed(bitstring)], dtype=np.float64)
            # Ising energy (minimize)
            e  = float(h @ bits)    # Note: in Ising {-1,+1}, but we mapped x∈{0,1}
            # Re-evaluate using QUBO objective (consistent)
            qobj = evaluate_qubo_vectorized(bits, lin, quad)
            energy += (-qobj) * (count / total)   # negate because QAOA minimizes

        iter_count[0] += 1
        if energy < best_energy[0]:
            best_energy[0]  = energy
            best_params[0]  = params.copy()

        return energy

    # ── Initial parameters ────────────────────────────────────────────────────
    rng    = np.random.default_rng(seed)
    x0     = rng.uniform(0, 2 * np.pi, n_params)

    # ── Run optimization ──────────────────────────────────────────────────────
    if opt_name.upper() == "COBYLA":
        result = scipy_minimize(
            objective, x0,
            method="COBYLA",
            options={"maxiter": max_iter, "rhobeg": 0.5},
        )
    else:
        result = scipy_minimize(
            objective, x0,
            method="SLSQP",
            options={"maxiter": max_iter},
        )

    # ── Sample final circuit ──────────────────────────────────────────────────
    opt_params = best_params[0] if best_params[0] is not None else result.x
    gammas_opt = opt_params[:p]
    betas_opt  = opt_params[p:]

    qc_final  = build_qaoa_circuit(h, J, p, gammas_opt, betas_opt)
    tqc_final = transpile(qc_final, simulator, optimization_level=0)
    job_final = simulator.run(tqc_final, shots=shots * 4, seed_simulator=seed + 1)
    counts    = job_final.result().get_counts()

    # ── Decode bitstrings and find best feasible ──────────────────────────────
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    best_mask       = None
    best_obj        = -np.inf
    n_feasible      = 0
    all_candidates  = []

    for bitstring, count in sorted_counts:
        bits = np.array(
            [int(b) for b in reversed(bitstring)], dtype=np.float64
        )
        obj      = evaluate_qubo_vectorized(bits, lin, quad)
        feasible = check_feasibility(bits, w, M)

        all_candidates.append({
            "mask":     bits,
            "obj":      obj,
            "count":    count,
            "feasible": feasible,
        })

        if feasible:
            n_feasible += 1
            if obj > best_obj:
                best_obj  = obj
                best_mask = bits.copy()

    # If no feasible candidate found, take the one closest to feasibility
    # (minimum memory violation) among the top-count bitstrings
    if best_mask is None:
        print("  [QAOA] WARNING: No feasible bitstring found. Selecting least-infeasible.")
        best_mask = _least_infeasible(all_candidates, w, M)
        best_obj  = evaluate_qubo_vectorized(best_mask, lin, quad)
        qaoa_feasible = False
    else:
        qaoa_feasible = True

    runtime_s = time.time() - t_start
    retained  = mask_to_channel_list(best_mask)
    pruned    = [i for i in range(num_channels) if i not in retained]

    # Circuit info
    qc_info   = build_qaoa_circuit(h, J, p, gammas_opt, betas_opt)
    circuit_info = {
        "num_qubits": qc_info.num_qubits,
        "depth":      qc_info.depth(),
        "num_gates":  qc_info.size(),
    }

    print(f"  [QAOA] p={p} | shots={shots} | optimizer={opt_name}")
    print(f"  [QAOA] Iterations : {iter_count[0]}")
    print(f"  [QAOA] Best mask  : {best_mask.astype(int).tolist()}")
    print(f"  [QAOA] Objective  : {best_obj:.6f}")
    print(f"  [QAOA] Retained   : {retained}")
    print(f"  [QAOA] Feasible   : {qaoa_feasible}")
    print(f"  [QAOA] Runtime    : {runtime_s:.2f}s")

    return {
        "method":             "qaoa",
        "best_mask":          best_mask,
        "best_objective":     best_obj,
        "retained_channels":  retained,
        "pruned_channels":    pruned,
        "runtime_s":          runtime_s,
        "feasible":           qaoa_feasible,
        "n_feasible_samples": n_feasible,
        "qaoa_depth":         p,
        "qaoa_shots":         shots,
        "qaoa_optimizer":     opt_name,
        "qaoa_iterations":    iter_count[0],
        "circuit_info":       circuit_info,
        "all_counts":         dict(sorted_counts[:20]),   # top-20 bitstrings
    }


def _least_infeasible(
    candidates: list,
    w: np.ndarray,
    M: float,
) -> np.ndarray:
    """Select the candidate with smallest memory constraint violation."""
    best     = None
    best_vio = np.inf
    for c in candidates:
        vio = max(0.0, float(np.dot(w, c["mask"])) - M)
        if vio < best_vio:
            best_vio = vio
            best     = c["mask"].copy()
    return best if best is not None else candidates[0]["mask"].copy()


def _qaoa_fallback(
    lin:  np.ndarray,
    quad: np.ndarray,
    w:    np.ndarray,
    M:    float,
    cfg:  Dict[str, Any],
) -> Dict[str, Any]:
    """
    Classical simulation fallback when Qiskit is unavailable.

    Simulates QAOA-like behavior by:
    1. Generating many random bitstrings (Monte Carlo sampling).
    2. Filtering for feasibility.
    3. Returning the best feasible solution.

    This is explicitly labeled as a CLASSICAL FALLBACK — NOT quantum execution.
    All results clearly state that quantum hardware was not used.
    """
    print("  [QAOA] WARNING: Running CLASSICAL FALLBACK (Qiskit unavailable)")
    print("  [QAOA] Results labeled as QAOA_FALLBACK, not true quantum execution.")

    t_start      = time.time()
    n            = len(lin)
    shots        = cfg.get("qaoa_shots", 512) * 4
    seed         = cfg.get("seed",       42)
    max_iter     = cfg.get("qaoa_max_iterations", 100)
    p            = cfg.get("qaoa_depth", 1)

    rng = np.random.default_rng(seed)

    best_mask = None
    best_obj  = -np.inf

    for _ in range(shots):
        bits = rng.integers(0, 2, size=n).astype(np.float64)
        if check_feasibility(bits, w, M):
            obj = evaluate_qubo_vectorized(bits, lin, quad)
            if obj > best_obj:
                best_obj  = obj
                best_mask = bits.copy()

    if best_mask is None:
        # All random samples infeasible → greedy fallback
        order = np.argsort(-lin)
        best_mask = np.zeros(n)
        used = 0.0
        for i in order:
            if used + w[i] <= M:
                best_mask[i] = 1.0
                used         += w[i]
        best_obj = evaluate_qubo_vectorized(best_mask, lin, quad)

    runtime_s = time.time() - t_start
    retained  = mask_to_channel_list(best_mask)
    pruned    = [i for i in range(n) if i not in retained]

    return {
        "method":             "qaoa_fallback",
        "best_mask":          best_mask,
        "best_objective":     best_obj,
        "retained_channels":  retained,
        "pruned_channels":    pruned,
        "runtime_s":          runtime_s,
        "feasible":           check_feasibility(best_mask, w, M),
        "n_feasible_samples": shots,
        "qaoa_depth":         p,
        "qaoa_shots":         shots,
        "qaoa_optimizer":     "CLASSICAL_FALLBACK",
        "qaoa_iterations":    max_iter,
        "circuit_info":       {"note": "Classical fallback — no quantum circuit"},
        "warning":            "Qiskit not available. Classical random search used.",
    }
