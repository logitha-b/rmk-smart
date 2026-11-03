"""Tests for model construction and parameter counting."""

import pytest
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn          import SmallCNN, build_model
from models.pruned_model import PrunedCNN, build_pruned_model
from models.model_utils  import (
    count_parameters,
    model_memory_bytes,
    model_memory_mb,
    verify_model_shapes,
)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def baseline_model(device):
    m = build_model(num_classes=100).to(device)
    return m


class TestSmallCNN:
    def test_output_shape(self, baseline_model, device):
        """Output should be (B, 100)."""
        x   = torch.randn(4, 3, 32, 32).to(device)
        out = baseline_model(x)
        assert out.shape == (4, 100), f"Expected (4, 100), got {out.shape}"

    def test_conv1_channels(self, baseline_model):
        """conv1 must have exactly 16 output channels."""
        assert baseline_model.conv1.out_channels == 16

    def test_conv1_in_channels(self, baseline_model):
        """conv1 must receive 3 input channels (RGB)."""
        assert baseline_model.conv1.in_channels == 3

    def test_forward_masked(self, baseline_model, device):
        """forward_masked should produce valid output with a binary mask."""
        x    = torch.randn(2, 3, 32, 32).to(device)
        mask = torch.ones(16)
        mask[0] = 0.0
        out  = baseline_model.forward_masked(x, mask)
        assert out.shape == (2, 100)

    def test_parameter_count_positive(self, baseline_model):
        n = count_parameters(baseline_model)
        assert n > 0, "Parameter count must be positive"

    def test_shape_verification(self, baseline_model, device):
        ok = verify_model_shapes(baseline_model, device)
        assert ok, "Shape verification failed"


class TestPrunedCNN:
    def test_output_shape_pruned(self, baseline_model, device):
        """Pruned model output should still be (B, 100)."""
        retained = [0, 2, 4, 6, 8, 10]
        pruned   = build_pruned_model(baseline_model, retained, num_classes=100).to(device)
        x   = torch.randn(4, 3, 32, 32).to(device)
        out = pruned(x)
        assert out.shape == (4, 100)

    def test_fewer_parameters_when_pruned(self, baseline_model, device):
        """Pruned model must have fewer parameters than baseline."""
        retained = [0, 2, 4, 6]   # keep only 4 of 16
        pruned   = build_pruned_model(baseline_model, retained, num_classes=100).to(device)
        assert count_parameters(pruned) < count_parameters(baseline_model)

    def test_pruned_conv1_channels(self, baseline_model, device):
        """Pruned conv1 must have k output channels."""
        retained = [1, 3, 5, 7, 9]
        pruned   = build_pruned_model(baseline_model, retained).to(device)
        assert pruned.conv1.out_channels == len(retained)

    def test_pruned_conv2_in_channels(self, baseline_model, device):
        """Pruned conv2 must accept k input channels."""
        retained = [0, 1, 2, 3]
        pruned   = build_pruned_model(baseline_model, retained).to(device)
        assert pruned.conv2.in_channels == len(retained)

    def test_weight_transfer(self, baseline_model, device):
        """Transferred conv1 weights must match the baseline for retained channels."""
        retained = [0, 5, 10]
        pruned   = build_pruned_model(baseline_model, retained).to(device)
        for new_idx, orig_idx in enumerate(retained):
            diff = (pruned.conv1.weight[new_idx] - baseline_model.conv1.weight[orig_idx]).abs().max()
            assert diff < 1e-6, f"Weight mismatch at channel {orig_idx}"

    def test_memory_less_than_baseline(self, baseline_model, device):
        """Pruned model memory must be less than baseline."""
        retained = [0, 1, 2, 3, 4, 5, 6, 7]   # 8 of 16
        pruned   = build_pruned_model(baseline_model, retained).to(device)
        assert model_memory_bytes(pruned) < model_memory_bytes(baseline_model)
