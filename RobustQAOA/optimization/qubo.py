"""
QUBO (Quadratic Unconstrained Binary Optimization) formulation.

Objective (maximization):
    maximize:
        α Σ c_i x_i
      + β Σ r_i x_i
      + γ Σ_{i<j} Q_ij x_i x_j
      - λ (Σ w_i x_i - M)^2

Binary variables:
    x_i ∈ {0, 1}   where x_i = 1 means channel i is retained.

Sign convention:
    This is a MAXIMIZATION problem.
    For QUBO solvers that minimize, we negate the objective.
    The QAOA cost Hamiltonian is constructed to be MINIMIZED;
    the sign is negated before passing to the circuit.

    The exact solver and the objective evaluation always return the
    MAXIMIZATION value (higher = better).

Penalty term expansion:
    (Σ w_i x_i - M)^2
    = Σ_i w_i^2 x_i^2
      + 2 Σ_{i<j} w_i w_j x_i x_j
      - 2M Σ_i w_i x_i
      + M^2

    For binary variables, x_i^2 = x_i, so:
    = Σ_i w_i^2 x_i
      + 2 Σ_{i<j} w_i w_j x_i x_j
      - 2M Σ_i w_i x_i
      + M^2

    The constant M^2 is dropped from the optimization objective
    (it does not affect the optimal bitstring).

    Linear coefficient:   linear_penalty_i = -λ (w_i^2 - 2M w_i)
    Quadratic coefficient: quad_penalty_ij  = -λ 2 w_i w_j   (i < j)
"""

import numpy as np
from typing import Dict, Any, Tuple


def build_qubo_matrix(
    importance: np.ndarray,
    robustness: np.ndarray,
    synergy:    np.ndarray,
    w:          np.ndarray,
    M:          float,
    alpha:      float,
    beta:       float,
    gamma:      float,
    lam:        float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the QUBO coefficient arrays.

    The objective to MAXIMIZE is:
        Σ_i lin[i] x_i  +  Σ_{i<j} quad[i,j] x_i x_j

    Args:
        importance : c_i array, shape (n,)
        robustness : r_i array, shape (n,)
        synergy    : Q_ij matrix, shape (n, n), symmetric, zero diagonal
        w          : memory cost array, shape (n,)
        M          : memory budget (bytes)
        alpha      : weight for clean performance
        beta       : weight for robustness
        gamma      : weight for pairwise synergy
        lam        : quadratic penalty coefficient

    Returns:
        lin  : linear  coefficients, shape (n,)     — for Σ lin[i] x_i
        quad : quadratic coefficients, shape (n, n) — for Σ quad[i,j] x_i x_j
               (only upper triangle i<j is meaningful; diagonal is 0)
    """
    n = len(importance)

    # ── Linear term ───────────────────────────────────────────────────────────
    # From objective: α c_i + β r_i
    lin = alpha * importance + beta * robustness

    # Penalty linear contribution: -λ (w_i^2 - 2M w_i)
    lin_penalty = -lam * (w**2 - 2.0 * M * w)
    lin = lin + lin_penalty

    # ── Quadratic term ────────────────────────────────────────────────────────
    # From synergy: γ Q_ij   (upper triangle only; diagonal = 0)
    quad = gamma * synergy.copy()

    # Penalty quadratic contribution: -λ * 2 * w_i * w_j   (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            penalty_ij   = -lam * 2.0 * w[i] * w[j]
            quad[i, j]  += penalty_ij
            quad[j, i]  += penalty_ij   # keep symmetric

    return lin, quad


def evaluate_qubo(
    mask:  np.ndarray,
    lin:   np.ndarray,
    quad:  np.ndarray,
) -> float:
    """
    Evaluate the QUBO objective for a binary mask (maximization value).

    Obj = Σ_i lin[i] x_i  +  Σ_{i<j} quad[i,j] x_i x_j

    Args:
        mask : binary array, shape (n,)
        lin  : linear coefficients
        quad : quadratic coefficients (symmetric matrix)

    Returns:
        objective value (float), higher is better.
    """
    obj = float(np.dot(lin, mask))
    n   = len(mask)
    for i in range(n):
        for j in range(i + 1, n):
            obj += float(quad[i, j] * mask[i] * mask[j])
    return obj


def evaluate_qubo_vectorized(
    mask: np.ndarray,
    lin:  np.ndarray,
    quad: np.ndarray,
) -> float:
    """
    Vectorized QUBO evaluation (faster for batch use in exact solver).

    Uses: obj = x^T * lin  +  0.5 * x^T * Q_sym * x - 0.5 * x^T * diag(Q) * x
    For binary x (x_i^2 = x_i) and symmetric Q with zero diagonal this simplifies to:
        obj = lin · x  +  Σ_{i<j} quad[i,j] * x_i * x_j

    Since quad is symmetric: Σ_{i<j} quad[i,j] x_i x_j = 0.5 * x^T Q x
    (because Q[i,j] = Q[j,i] and Q[i,i] = 0)
    """
    linear_part = float(np.dot(lin, mask))
    # 0.5 * x^T Q x — correct for symmetric Q with zero diagonal
    quad_part   = 0.5 * float(mask @ quad @ mask)
    return linear_part + quad_part


def qubo_to_ising(
    lin:  np.ndarray,
    quad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert QUBO coefficients to Ising (Pauli Z) form for QAOA.

    QUBO: min  Σ lin[i] x_i  + Σ_{i<j} quad[i,j] x_i x_j
    (Note: for QAOA we minimize, so negate the maximization objective)

    Ising transformation: x_i = (1 - z_i) / 2,  z_i ∈ {-1, +1}

    After substitution:
        h[i]   : linear (field) coefficient on Z_i
        J[i,j] : quadratic coupling coefficient on Z_i Z_j
        offset : constant term

    Sign for QAOA:
        We want to MAXIMIZE the QUBO objective.
        QAOA minimizes the cost Hamiltonian.
        Therefore: H_C = -(QUBO objective) expressed in Ising form.

    Returns:
        h      : Ising field coefficients (n,) — for Z_i terms in H_C
        J      : Ising coupling coefficients (n, n) — for Z_i Z_j terms
        offset : constant offset
    """
    n = len(lin)

    # We minimize the NEGATED QUBO: -( Σ lin_i x_i + Σ_{i<j} quad_ij x_i x_j )
    # With x_i = (1 - z_i)/2:
    #   x_i = 0.5 - 0.5 z_i
    #   x_i x_j = (1 - z_i)(1 - z_j)/4 = 0.25(1 - z_i - z_j + z_i z_j)

    neg_lin  = -lin
    neg_quad = -quad

    h      = np.zeros(n,        dtype=np.float64)
    J      = np.zeros((n, n),   dtype=np.float64)
    offset = 0.0

    # Linear terms: neg_lin_i * x_i = neg_lin_i * (0.5 - 0.5 z_i)
    for i in range(n):
        offset += 0.5 * neg_lin[i]
        h[i]   -= 0.5 * neg_lin[i]

    # Quadratic terms: neg_quad_ij * x_i x_j = neg_quad_ij * 0.25(1 - z_i - z_j + z_i z_j)
    for i in range(n):
        for j in range(i + 1, n):
            q = neg_quad[i, j]
            offset += 0.25 * q
            h[i]   -= 0.25 * q
            h[j]   -= 0.25 * q
            J[i, j] += 0.25 * q
            J[j, i] += 0.25 * q

    return h, J, offset


def print_qubo_summary(lin: np.ndarray, quad: np.ndarray) -> None:
    """Print a summary of QUBO coefficients."""
    print(f"  [QUBO] Linear  range : [{lin.min():.4f},  {lin.max():.4f}]")
    # Only upper triangle
    upper = np.array([quad[i, j] for i in range(len(lin)) for j in range(i+1, len(lin))])
    if len(upper) > 0:
        print(f"  [QUBO] Quadratic range: [{upper.min():.4f}, {upper.max():.4f}]")
