"""
Fast Gradient Sign Method (FGSM) adversarial attack.

Formula:
    x_adv = x + epsilon * sign(∇_x L(f(x), y))

where:
    x       : clean input
    y       : true label
    L       : cross-entropy loss
    epsilon : perturbation magnitude

The adversarial example is clipped to the valid normalised input range
[x_min, x_max] derived from the CIFAR-100 normalisation statistics.
"""

import torch
import torch.nn as nn
from typing import Tuple


# CIFAR-100 per-channel pixel range after normalisation
# Original [0,1] → normalised: (0 - mean) / std ... (1 - mean) / std
_CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
_CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761])


def _input_range() -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel (min, max) after CIFAR-100 normalisation."""
    lo = (0.0 - _CIFAR100_MEAN) / _CIFAR100_STD
    hi = (1.0 - _CIFAR100_MEAN) / _CIFAR100_STD
    return lo, hi


def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate FGSM adversarial examples.

    Args:
        model   : model to attack (used for gradient computation)
        inputs  : clean inputs, shape (B, C, H, W)
        targets : true labels, shape (B,)
        epsilon : perturbation magnitude (in normalised space)
        device  : torch device

    Returns:
        adv_inputs : adversarial examples, same shape as inputs, clipped to valid range
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    inputs_  = inputs.clone().detach().to(device).requires_grad_(True)
    targets_ = targets.to(device)

    outputs = model(inputs_)
    loss    = criterion(outputs, targets_)
    model.zero_grad()
    loss.backward()

    # FGSM perturbation
    perturbation = epsilon * inputs_.grad.sign()
    adv_inputs   = inputs_ + perturbation

    # Clip to valid input range (per-channel)
    lo, hi = _input_range()
    lo = lo.view(1, 3, 1, 1).to(device)
    hi = hi.view(1, 3, 1, 1).to(device)
    adv_inputs = torch.clamp(adv_inputs, min=lo, max=hi)

    return adv_inputs.detach()


def fgsm_accuracy(
    model: nn.Module,
    loader,
    epsilon: float,
    device: torch.device,
    n_samples: int = None,
) -> float:
    """
    Evaluate model accuracy under FGSM attack.

    Args:
        model     : model to evaluate
        loader    : DataLoader
        epsilon   : FGSM epsilon
        device    : torch device
        n_samples : if set, only evaluate this many samples

    Returns:
        Adversarial accuracy (float in [0, 1]).
    """
    model.eval()
    correct, total = 0, 0

    for inputs, targets in loader:
        if n_samples and total >= n_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        adv = fgsm_attack(model, inputs, targets, epsilon, device)

        with torch.no_grad():
            out  = model(adv)
            pred = out.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total   += inputs.size(0)

    return correct / total if total > 0 else 0.0
