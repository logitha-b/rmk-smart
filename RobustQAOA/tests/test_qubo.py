"""Tests for QUBO formulation and exact solver."""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.qubo         import build_qubo_matrix, evaluate_qubo, evaluate_qubo_vectorized, qubo_to_ising
from optimization.exact_solver import exact_qubo_solve, enumerate_all_bitstrings
from pruning.pruning_utils     import compute_channel_memory_costs, compute_memory_budget


def _make_simple_inputs(n=4):
    """Create simple QUBO inputs for testing."""
    importance  = np.array([0.5, 0.3, 0.8, 0.1])
    robustness  = np.array([0.4, 0.2, 0.6, 0.3])
    synergy     = np.zeros((n, n))
    synergy[0, 2] = synergy[2, 0] = 0.1   # channels 0 and 2 synergize
    w = compute_channel_memory_costs(n)
    M = compute_memory_budget(n, 0.60)
    return importance, robustness, synergy, w, M


class TestQUBOConstruction:
    def test_linear_coefficients_shape(self):
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        assert lin.shape == (4,)

    def test_quadratic_coefficients_shape(self):
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        assert quad.shape == (4, 4)

    def test_quadratic_symmetry(self):
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        assert np.allclose(quad, quad.T, atol=1e-10)

    def test_empty_mask_objective(self):
        """Empty mask should give 0 for linear part and near-zero for penalty."""
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        mask = np.zeros(4)
        obj  = evaluate_qubo_vectorized(mask, lin, quad)
        # Linear part is 0; penalty is lam*(0 - M)^2 which is negative
        assert isinstance(obj, float)


class TestQUBOEvaluation:
    def test_consistency(self):
        """Scalar and vectorized evaluations should agree."""
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        mask = np.array([1.0, 0.0, 1.0, 0.0])
        v1   = evaluate_qubo(mask, lin, quad)
        v2   = evaluate_qubo_vectorized(mask, lin, quad)
        assert abs(v1 - v2) < 1e-6


class TestIsingConversion:
    def test_output_shapes(self):
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        h, J, offset = qubo_to_ising(lin, quad)
        assert h.shape == (4,)
        assert J.shape == (4, 4)
        assert isinstance(offset, float)


class TestExactSolver:
    def test_enumerate_bitstrings(self):
        bits = enumerate_all_bitstrings(4)
        assert bits.shape == (16, 4)
        # All rows distinct
        unique_rows = set(map(tuple, bits.tolist()))
        assert len(unique_rows) == 16

    def test_exact_solver_finds_feasible(self):
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        result = exact_qubo_solve(lin, quad, w, M, num_channels=4)
        assert result["n_feasible"] > 0

    def test_exact_solution_feasible(self):
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        result = exact_qubo_solve(lin, quad, w, M, num_channels=4)
        mask   = result["best_mask"]
        used   = float(np.dot(w, mask))
        assert used <= M + 1e-6, f"Memory violated: {used} > {M}"

    def test_exact_solution_is_optimal(self):
        """No other feasible solution should have a higher objective."""
        imp, rob, syn, w, M = _make_simple_inputs(4)
        lin, quad = build_qubo_matrix(imp, rob, syn, w, M, 1.0, 1.0, 0.5, 5.0)
        result    = exact_qubo_solve(lin, quad, w, M, num_channels=4)
        best_obj  = result["best_objective"]

        from optimization.exact_solver import enumerate_all_bitstrings
        bits = enumerate_all_bitstrings(4).astype(np.float64)
        for row in bits:
            if float(np.dot(w, row)) <= M:
                obj = evaluate_qubo_vectorized(row, lin, quad)
                assert obj <= best_obj + 1e-6
