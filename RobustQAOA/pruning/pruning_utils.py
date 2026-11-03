"""
Channel memory cost calculation and feasibility utilities.

Memory cost per channel (w_i):
    The memory cost of retaining channel i in conv1 is the number of bytes
    required to store the weights associated with that channel in:
      1. conv1    : 1 filter of shape (in_channels, kH, kW) = (3, 3, 3) = 27 floats
      2. bn1      : 1 weight + 1 bias + 1 running_mean + 1 running_var = 4 floats
      3. conv2    : The entire conv2 is affected because its input feature maps
                    have fewer channels.  The cost attributable to channel i in
                    conv2 is: 1 input-channel slice across all 32 filters
                    = 32 × kH × kW = 32 × 3 × 3 = 288 floats.

    Total floats for channel i = 27 + 4 + 288 = 319 floats
    Bytes per channel = 319 × 4 (float32) = 1276 bytes

    Because all 16 channels are identical in structure, w_i is the same for
    every channel.  This is derived directly from the model architecture —
    not assigned arbitrarily.

Memory budget M:
    M = memory_budget_fraction × (16 × w_i)
    i.e., the user specifies a fraction of the full-layer budget.

Constraint:
    Σ w_i x_i ≤ M
"""

import numpy as np
import torch.nn as nn
from typing import List, Dict


# ── Per-channel cost constants (derived from SmallCNN architecture) ──────────
# conv1 filter  : 3 × 3 × 3 = 27 parameters
# bn1 params    : weight(1) + bias(1) + running_mean(1) + running_var(1) = 4
# conv2 slice   : 32 filters × 1 in-channel × 3 × 3 = 288 parameters
_CONV1_FILTER_PARAMS = 3 * 3 * 3          # = 27
_BN1_PARAMS_PER_CH   = 4                  # weight, bias, r_mean, r_var
_CONV2_SLICE_PARAMS  = 32 * 3 * 3         # = 288
_BYTES_PER_FLOAT32   = 4

PARAMS_PER_CHANNEL = _CONV1_FILTER_PARAMS + _BN1_PARAMS_PER_CH + _CONV2_SLICE_PARAMS
# = 27 + 4 + 288 = 319 params/channel
BYTES_PER_CHANNEL  = PARAMS_PER_CHANNEL * _BYTES_PER_FLOAT32
# = 319 × 4 = 1276 bytes/channel


def compute_channel_memory_costs(num_channels: int = 16) -> np.ndarray:
    """
    Return an array w of shape (num_channels,) where w[i] = memory cost in bytes
    of retaining channel i.

    Since all channels have identical architecture, w[i] is constant.

    Returns:
        w : np.ndarray of shape (num_channels,), dtype float64
    """
    w = np.full(num_channels, float(BYTES_PER_CHANNEL), dtype=np.float64)
    return w


def compute_memory_budget(
    num_channels: int,
    budget_fraction: float,
) -> float:
    """
    Compute the memory budget M in bytes.

    M = budget_fraction × (num_channels × BYTES_PER_CHANNEL)

    Args:
        num_channels    : total number of prunable channels (16)
        budget_fraction : fraction in (0, 1], e.g. 0.60

    Returns:
        M : float, memory budget in bytes
    """
    total_bytes = num_channels * BYTES_PER_CHANNEL
    M = budget_fraction * total_bytes
    return M


def check_feasibility(
    mask: np.ndarray,
    w: np.ndarray,
    M: float,
) -> bool:
    """
    Check whether a binary selection mask satisfies the memory constraint.

    Args:
        mask : binary array of shape (num_channels,), values in {0, 1}
        w    : memory cost array of shape (num_channels,)
        M    : memory budget (bytes)

    Returns:
        True if Σ w_i x_i ≤ M, False otherwise.
    """
    used = float(np.dot(w, mask))
    return used <= M


def mask_to_channel_list(mask: np.ndarray) -> List[int]:
    """Convert binary mask to list of retained channel indices."""
    return [int(i) for i, v in enumerate(mask) if v > 0.5]


def channel_list_to_mask(channels: List[int], num_channels: int = 16) -> np.ndarray:
    """Convert list of retained channel indices to binary mask."""
    mask = np.zeros(num_channels, dtype=np.float64)
    for c in channels:
        mask[c] = 1.0
    return mask


def print_memory_info(num_channels: int = 16, budget_fraction: float = 0.60) -> None:
    """Print a summary of the memory cost model."""
    w = compute_channel_memory_costs(num_channels)
    M = compute_memory_budget(num_channels, budget_fraction)
    print(f"  [memory] Params/channel          : {PARAMS_PER_CHANNEL}")
    print(f"  [memory] Bytes/channel            : {BYTES_PER_CHANNEL}")
    print(f"  [memory] Total layer cost (16ch)  : {num_channels * BYTES_PER_CHANNEL} bytes")
    print(f"  [memory] Budget ({budget_fraction*100:.0f}%)          : {M:.0f} bytes "
          f"(≤ {M / BYTES_PER_CHANNEL:.1f} channels)")
    max_channels = int(M / BYTES_PER_CHANNEL)
    print(f"  [memory] Max retained channels    : {max_channels}")
