from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def mse(prediction: Array, target: Array) -> tuple[float, Array]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} must equal target shape {target.shape}")
    difference = prediction - target
    gradient = np.asarray(2.0 * difference / difference.size, dtype=np.float64)
    return float(np.mean(difference**2)), gradient


def binary_cross_entropy_from_logits(logits: Array, targets: Array) -> tuple[float, Array]:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if np.any((targets < 0) | (targets > 1)):
        raise ValueError("binary targets must be in [0, 1]")
    losses = np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    probabilities = np.empty_like(logits)
    positive = logits >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    gradient = np.asarray((probabilities - targets) / targets.size, dtype=np.float64)
    return float(np.mean(losses)), gradient


def softmax_cross_entropy(logits: Array, target_index: int) -> tuple[float, Array, Array]:
    flat = logits.reshape(-1)
    if not 0 <= target_index < flat.size:
        raise ValueError("target index is outside the class range")
    shifted = flat - np.max(flat)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probabilities = shifted - log_sum_exp
    probabilities = np.exp(log_probabilities)
    gradient = probabilities.copy()
    gradient[target_index] -= 1.0
    return float(-log_probabilities[target_index]), gradient.reshape(logits.shape), probabilities
