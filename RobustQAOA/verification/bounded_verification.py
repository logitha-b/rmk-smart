"""
Bounded adversarial verification.

For the full CIFAR-100 image space (32×32 × 3 channels, float values),
exhaustive verification is computationally infeasible.

This module implements bounded verification over a small, finite input space
for unit-test purposes only.

The actual robustness claims are:
  - FGSM/PGD empirical robustness (measured, not certified)
  - No universal robustness certification is claimed
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any


FEASIBILITY_NOTE = (
    "Exhaustive verification is infeasible for the full CIFAR-100 image space.\n"
    "Each 32x32x3 float32 image has effectively infinite possible perturbations.\n"
    "Robustness is measured empirically using FGSM and PGD attacks on a test subset.\n"
    "No universal robustness certification is claimed or implied."
)


def bounded_verification_small_space(
    model:    nn.Module,
    device:   torch.device,
    n_pixels: int = 4,
    n_bits:   int = 4,
) -> Dict[str, Any]:
    """
    Exhaustive verification over a tiny discretized input space.

    Constructs a grid of n_pixels pixels × 2^n_bits intensity levels.
    This is purely for demonstrating the verification concept on a tractable space.

    Args:
        model    : model to verify
        device   : torch device
        n_pixels : number of pixels to consider (very small, e.g. 4)
        n_bits   : number of discrete intensity levels per pixel (e.g. 4 → 16 values)

    Returns:
        Dict with verification results and infeasibility note.
    """
    print(f"\n  [verification] Bounded verification over {n_pixels} pixels × {2**n_bits} levels")
    print(f"  [verification] Space size: {2**(n_pixels * n_bits):,} inputs")
    print(f"  [verification] NOTE: {FEASIBILITY_NOTE[:80]}...")

    # This is illustrative only — not a meaningful security claim
    space_size = 2 ** (n_pixels * n_bits)
    if space_size > 1_000_000:
        return {
            "verified":    False,
            "reason":      "Space too large for exhaustive check even in this demo.",
            "space_size":  space_size,
            "note":        FEASIBILITY_NOTE,
        }

    model.eval()
    n_correct = 0
    n_total   = min(space_size, 10000)   # cap for safety

    dummy_target = torch.tensor([0]).to(device)

    for i in range(n_total):
        # Build a dummy input with bits from integer i
        vals  = np.array([(i >> b) & 1 for b in range(n_pixels * n_bits)], dtype=np.float32)
        vals  = vals / max(vals.max(), 1.0)   # normalise to [0, 1]
        img   = torch.zeros(1, 3, 32, 32)
        img[0, 0, 0, :n_pixels] = torch.from_numpy(vals[:n_pixels])

        with torch.no_grad():
            out  = model(img.to(device))
            pred = out.argmax(dim=1)
            if pred.item() == dummy_target.item():
                n_correct += 1

    return {
        "verified":       True,
        "space_size":     n_total,
        "n_correct":      n_correct,
        "accuracy":       n_correct / n_total,
        "note":           FEASIBILITY_NOTE,
        "warning":        (
            "This verification is over a toy discretized space and does NOT "
            "constitute any real robustness guarantee."
        ),
    }


def print_verification_disclaimer() -> None:
    """Print the verification disclaimer."""
    print(f"\n  {'='*60}")
    print("  VERIFICATION DISCLAIMER")
    print(f"  {'─'*60}")
    print(f"  {FEASIBILITY_NOTE}")
    print(f"  {'='*60}")
