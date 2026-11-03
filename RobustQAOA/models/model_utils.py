"""
Model utility functions: parameter counting, memory measurement, inference timing.
"""

import time
import torch
import torch.nn as nn
from typing import Tuple


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return total number of parameters (including non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def model_memory_bytes(model: nn.Module) -> int:
    """
    Estimate model memory usage in bytes.

    Sums up the number of elements × bytes-per-element for every parameter tensor.
    This reflects the actual storage cost of the model weights.

    Returns:
        Memory in bytes (int).
    """
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes


def model_memory_mb(model: nn.Module) -> float:
    """Return model memory in megabytes (float)."""
    return model_memory_bytes(model) / (1024 ** 2)


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = 10,
    repetitions: int = 100,
) -> float:
    """
    Measure average per-sample inference time in milliseconds.

    Args:
        model       : model in eval mode
        device      : torch device
        input_shape : shape of a single inference batch
        warmup      : number of warmup forward passes
        repetitions : number of timed repetitions

    Returns:
        Average inference time in milliseconds (float).
    """
    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    # Timed runs
    if device.type == 'cuda':
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(repetitions):
                _ = model(dummy)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / repetitions
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(repetitions):
                _ = model(dummy)
        elapsed_ms = (time.perf_counter() - start) * 1000 / repetitions

    return elapsed_ms


def save_checkpoint(model: nn.Module, path: str, metadata: dict = None) -> None:
    """Save model state dict with optional metadata."""
    payload = {"state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)
    print(f"  [checkpoint] Saved to {path}")


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    """Load model state dict from checkpoint. Returns metadata dict (may be empty)."""
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})


def verify_model_shapes(model: nn.Module, device: torch.device) -> bool:
    """
    Run a single forward pass with a dummy input to verify tensor shapes.

    Returns True if the forward pass succeeds without errors.
    """
    model.eval()
    try:
        dummy = torch.randn(2, 3, 32, 32).to(device)
        with torch.no_grad():
            out = model(dummy)
        expected_shape = (2, 100)
        assert out.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {out.shape}"
        )
        return True
    except Exception as e:
        print(f"  [shape check] FAILED: {e}")
        return False
