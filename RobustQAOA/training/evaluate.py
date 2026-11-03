"""
Evaluation routines for CIFAR-100 models.

Computes:
  - accuracy
  - loss
  - macro precision
  - macro recall
  - macro F1
  - parameter count
  - memory (MB)
  - inference time (ms)
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Any


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module = None,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a DataLoader.

    Returns a dict with keys:
        loss, accuracy, precision, recall, f1
    """
    if device is None:
        device = next(model.parameters()).device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss    = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            predicted   = outputs.argmax(dim=1)
            correct    += predicted.eq(targets).sum().item()
            total      += inputs.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    avg_loss  = total_loss / total
    accuracy  = correct    / total
    precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    recall    = recall_score(all_targets, all_preds,    average="macro", zero_division=0)
    f1        = f1_score(all_targets, all_preds,        average="macro", zero_division=0)

    return {
        "loss":      avg_loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "n_samples": total,
    }


def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Full evaluation including model statistics.

    Returns all metrics required by the benchmark.
    """
    from models.model_utils import (
        count_parameters,
        model_memory_mb,
        measure_inference_time,
    )

    criterion = nn.CrossEntropyLoss()
    metrics   = evaluate_model(model, loader, criterion, device)

    n_params   = count_parameters(model)
    mem_mb     = model_memory_mb(model)
    infer_ms   = measure_inference_time(model, device)

    metrics.update({
        "parameters":    n_params,
        "memory_mb":     mem_mb,
        "inference_ms":  infer_ms,
    })
    return metrics


def evaluate_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate on a fixed random subset of the loader.

    Useful for expensive ablation studies.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    rng = np.random.default_rng(seed)
    all_inputs, all_targets_list = [], []

    for inputs, targets in loader:
        all_inputs.append(inputs)
        all_targets_list.append(targets)

    all_inputs_t  = torch.cat(all_inputs,        dim=0)
    all_targets_t = torch.cat(all_targets_list,  dim=0)

    total = all_inputs_t.size(0)
    n     = min(n_samples, total)
    idx   = rng.choice(total, size=n, replace=False)

    subset_inputs  = all_inputs_t[idx].to(device)
    subset_targets = all_targets_t[idx].to(device)

    total_loss, correct = 0.0, 0
    all_preds = []

    batch_size = 128
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end    = min(start + batch_size, n)
            inp    = subset_inputs[start:end]
            tgt    = subset_targets[start:end]
            out    = model(inp)
            loss   = criterion(out, tgt)
            total_loss += loss.item() * inp.size(0)
            pred   = out.argmax(dim=1)
            correct += pred.eq(tgt).sum().item()
            all_preds.extend(pred.cpu().numpy())

    all_preds   = np.array(all_preds)
    targets_np  = subset_targets.cpu().numpy()

    avg_loss  = total_loss / n
    accuracy  = correct    / n
    precision = precision_score(targets_np, all_preds, average="macro", zero_division=0)
    recall    = recall_score(targets_np, all_preds,    average="macro", zero_division=0)
    f1        = f1_score(targets_np, all_preds,        average="macro", zero_division=0)

    return {
        "loss":      avg_loss,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "n_samples": n,
    }
