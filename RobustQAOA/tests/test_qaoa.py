"""Tests for QAOA bitstring decoding and feasibility."""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.qubo import build_qubo_matrix, evaluate_qubo_vectorized
from optimization.qaoa  import _least_infeasible
from pruning.pruning_utils import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
    mask_to_channel_list,
)


def _make_qubo(n=4):
    importance  = np.ones(n) * 0.5
    robustness  = np.ones(n) * 0.3
    synergy     = np.zeros((n, n))
    w = compute_channel_memory_costs(n)
    M = compute_memory_budget(n, 0.60)
    lin, quad = build_qubo_matrix(importance, robustness, synergy, w, M,
                                   1.0, 1.0, 0.5, 5.0)
    return lin, quad, w, M


class TestBitstringDecoding:
    def test_mask_to_channel_list_correct(self):
        mask = np.array([1, 0, 1, 0], dtype=float)
        assert mask_to_channel_list(mask) == [0, 2]

    def test_all_zeros_empty_list(self):
        mask = np.zeros(4)
        assert mask_to_channel_list(mask) == []

    def test_all_ones_full_list(self):
        mask = np.ones(4)
        assert mask_to_channel_list(mask) == [0, 1, 2, 3]


class TestFeasibilityChecking:
    def test_feasible_mask(self):
        lin, quad, w, M = _make_qubo(4)
        mask = np.zeros(4)
        assert check_feasibility(mask, w, M)

    def test_infeasible_mask(self):
        lin, quad, w, M = _make_qubo(4)
        M_tight = 0.1  # very tight
        mask = np.ones(4)
        assert not check_feasibility(mask, w, M_tight)

    def test_least_infeasible_selection(self):
        lin, quad, w, M = _make_qubo(4)
        M_tight = 0.1
        candidates = [
            {"mask": np.array([1.0, 1.0, 1.0, 1.0]), "obj": 10.0, "count": 100, "feasible": False},
            {"mask": np.array([1.0, 0.0, 0.0, 0.0]), "obj": 2.0,  "count": 50,  "feasible": False},
            {"mask": np.array([0.0, 0.0, 0.0, 0.0]), "obj": 0.0,  "count": 10,  "feasible": True},
        ]
        best = _least_infeasible(candidates, w, M_tight)
        # Should be the one with least memory usage (empty mask = 0 bytes)
        assert best.sum() == 0.0


class TestQUBOObjective:
    def test_better_mask_higher_objective(self):
        """The mask with more high-importance channels should score better
        (ignoring the penalty)."""
        importance = np.array([0.8, 0.1, 0.8, 0.1])
        robustness = np.zeros(4)
        synergy    = np.zeros((4, 4))
        w          = compute_channel_memory_costs(4)
        M          = compute_memory_budget(4, 0.60)
        lin, quad  = build_qubo_matrix(importance, robustness, synergy, w, M,
                                        1.0, 0.0, 0.0, 0.0)  # no penalty

        good_mask = np.array([1.0, 0.0, 1.0, 0.0])
        bad_mask  = np.array([0.0, 1.0, 0.0, 1.0])
        assert evaluate_qubo_vectorized(good_mask, lin, quad) > \
               evaluate_qubo_vectorized(bad_mask, lin, quad)
