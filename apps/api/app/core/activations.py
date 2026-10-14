from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def activate(x: Array, name: str) -> Array:
    if name == "identity":
        return x.copy()
    if name == "relu":
        return np.asarray(np.maximum(x, 0.0), dtype=np.float64)
    if name == "tanh":
        return np.asarray(np.tanh(x), dtype=np.float64)
    if name == "sigmoid":
        out = np.empty_like(x)
        positive = x >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out
    raise ValueError(f"Unknown activation: {name}")


def activation_derivative(x: Array, name: str) -> Array:
    if name == "identity":
        return np.ones_like(x)
    if name == "relu":
        return (x > 0).astype(np.float64)  # documented subgradient at zero: 0
    if name == "tanh":
        value = np.tanh(x)
        return np.asarray(1.0 - value * value, dtype=np.float64)
    if name == "sigmoid":
        value = activate(x, "sigmoid")
        return np.asarray(value * (1.0 - value), dtype=np.float64)
    raise ValueError(f"Unknown activation: {name}")
