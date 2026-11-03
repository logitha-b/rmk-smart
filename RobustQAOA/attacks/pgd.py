"""
Projected Gradient Descent (PGD) adversarial attack.

PGD is an iterative extension of FGSM.
At each step:
    x^{t+1} = Π_{x + S} ( x^t + step_size * sign(∇_x L(f(x^t), y)) )

where:
    S        : ℓ∞ ball of radius epsilon around the clean input x
    Π_{x+S} : projection operator that clips x^{t+1} to [x - ε, x + ε]
               and then clips to the valid normalised range

Initialisation:
    x^0 = x + uniform_noise ~ Uniform[-epsilon, epsilon]  (random start)

This is the Madry et al. (2018) formulation.
"""

import torch
import torch.nn as nn
from typing import Optional


_CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
_CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761])


def _input_range(device):
    """Return per-channel (min, max) after CIFAR-100 normalisation."""
    lo = ((0.0 - _CIFAR100_MEAN) / _CIFAR100_STD).view(1, 3, 1, 1).to(device)
    hi = ((1.0 - _CIFAR100_MEAN) / _CIFAR100_STD).view(1, 3, 1, 1).to(device)
    return lo, hi


def pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    step_size: float,
    num_steps: int,
    device: torch.device,
    random_start: bool = True,
) -> torch.Tensor:
    """
    Generate PGD adversarial examples.

    Args:
        model       : model to attack
        inputs      : clean inputs, shape (B, C, H, W)
        targets     : true labels, shape (B,)
        epsilon     : maximum perturbation (ℓ∞ ball radius)
        step_size   : step size per iteration (alpha)
        num_steps   : number of PGD iterations
        device      : torch device
        random_start: whether to initialise with random perturbation

    Returns:
        adv_inputs : adversarial examples, clipped to valid range
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    inputs_  = inputs.clone().detach().to(device)
    targets_ = targets.to(device)

    lo_global, hi_global = _input_range(device)

    if random_start:
        # Random initialisation within ℓ∞ ball
        delta = torch.zeros_like(inputs_).uniform_(-epsilon, epsilon)
        adv   = torch.clamp(inputs_ + delta, min=lo_global, max=hi_global).detach()
    else:
        adv = inputs_.clone()

    for _ in range(num_steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss    = criterion(outputs, targets_)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Gradient sign step
            adv_next = adv + step_size * adv.grad.sign()

            # Project back into ℓ∞ ball around clean input
            adv_next = torch.clamp(adv_next, inputs_ - epsilon, inputs_ + epsilon)

            # Clip to valid normalised range
            adv_next = torch.clamp(adv_next, min=lo_global, max=hi_global)

        adv = adv_next.detach()

    return adv


def pgd_accuracy(
    model: nn.Module,
    loader,
    epsilon: float,
    step_size: float,
    num_steps: int,
    device: torch.device,
    n_samples: Optional[int] = None,
) -> float:
    """
    Evaluate model accuracy under PGD attack.

    Returns:
        Adversarial accuracy (float in [0, 1]).
    """
    model.eval()
    correct, total = 0, 0

    for inputs, targets in loader:
        if n_samples and total >= n_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)

        adv = pgd_attack(model, inputs, targets, epsilon, step_size, num_steps, device)

        with torch.no_grad():
            out  = model(adv)
            pred = out.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total   += inputs.size(0)

    return correct / total if total > 0 else 0.0
