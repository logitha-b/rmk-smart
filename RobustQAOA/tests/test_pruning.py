"""Tests for channel pruning utilities."""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.pruning_utils import (
    compute_channel_memory_costs,
    compute_memory_budget,
    check_feasibility,
    mask_to_channel_list,
    channel_list_to_mask,
    BYTES_PER_CHANNEL,
    PARAMS_PER_CHANNEL,
)


class TestMemoryCost:
    def test_params_per_channel(self):
        """319 params per channel: 27 (conv1) + 4 (bn1) + 288 (conv2 slice)."""
        assert PARAMS_PER_CHANNEL == 319

    def test_bytes_per_channel(self):
        """1276 bytes per channel: 319 * 4."""
        assert BYTES_PER_CHANNEL == 319 * 4

    def test_cost_array_shape(self):
        w = compute_channel_memory_costs(16)
        assert w.shape == (16,)

    def test_cost_array_values(self):
        w = compute_channel_memory_costs(16)
        assert (w == BYTES_PER_CHANNEL).all()

    def test_memory_budget(self):
        M = compute_memory_budget(16, 0.60)
        expected = 0.60 * 16 * BYTES_PER_CHANNEL
        assert abs(M - expected) < 1e-6


class TestFeasibility:
    def test_full_mask_feasible_at_1(self):
        """Retaining all channels should be feasible at budget=1.0."""
        w    = compute_channel_memory_costs(16)
        M    = compute_memory_budget(16, 1.0)
        mask = np.ones(16)
        assert check_feasibility(mask, w, M)

    def test_empty_mask_feasible(self):
        w    = compute_channel_memory_costs(16)
        M    = compute_memory_budget(16, 0.60)
        mask = np.zeros(16)
        assert check_feasibility(mask, w, M)

    def test_exceeds_budget(self):
        """Retaining all channels should violate a tight budget."""
        w    = compute_channel_memory_costs(16)
        M    = compute_memory_budget(16, 0.30)
        mask = np.ones(16)
        assert not check_feasibility(mask, w, M)

    def test_partial_mask_feasible(self):
        """Keeping half the channels should fit in 60% budget."""
        w       = compute_channel_memory_costs(16)
        M       = compute_memory_budget(16, 0.60)
        mask    = np.zeros(16)
        mask[:9] = 1.0   # 9 channels < 60% of 16 = 9.6
        assert check_feasibility(mask, w, M)


class TestMaskConversion:
    def test_mask_to_channel_list(self):
        mask = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        assert mask_to_channel_list(mask) == [0, 2, 4]

    def test_channel_list_to_mask(self):
        channels = [0, 5, 10]
        mask = channel_list_to_mask(channels, 16)
        assert mask[0] == 1.0
        assert mask[5] == 1.0
        assert mask[10] == 1.0
        assert mask.sum() == 3.0

    def test_roundtrip(self):
        original = [1, 3, 7, 12]
        mask     = channel_list_to_mask(original, 16)
        recovered = mask_to_channel_list(mask)
        assert recovered == original
