from __future__ import annotations

from typing import Any

import numpy as np

from app.core.tracing import TraceStep, trace_document


def train_linear(seed: int = 7, learning_rate: float = 0.1, epochs: int = 10, batch_size: int = 4) -> dict[str, Any]:
    if not 0 < learning_rate <= 10:
        raise ValueError("learning_rate must be in (0, 10]")
    if not 1 <= epochs <= 500 or not 1 <= batch_size <= 4:
        raise ValueError("epochs must be 1..500 and batch_size must be 1..4")
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = 2.0 * x + 1.0
    rng = np.random.default_rng(seed)
    weight = float(rng.normal(0, 0.2))
    bias = 0.0
    initial = {"weight": weight, "bias": bias}
    snapshots: list[dict[str, Any]] = []
    steps: list[TraceStep] = []
    status = "completed"
    for epoch in range(epochs):
        for start in range(0, len(x), batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
            prediction = xb * weight + bias
            error = prediction - yb
            loss = float(np.mean(error**2))
            d_weight = float(np.mean(2.0 * error * xb))
            d_bias = float(np.mean(2.0 * error))
            old_weight, old_bias = weight, bias
            candidate_weight = weight - learning_rate * d_weight
            candidate_bias = bias - learning_rate * d_bias
            if not np.isfinite([loss, candidate_weight, candidate_bias]).all() or abs(candidate_weight) > 1e12:
                status = "diverged_safely"
                break
            weight, bias = candidate_weight, candidate_bias
            snapshot = {"epoch": epoch + 1, "batch_start": start, "loss": loss, "weight": weight, "bias": bias, "gradient_weight": d_weight, "gradient_bias": d_bias}
            snapshots.append(snapshot)
            step_id = f"update-{len(snapshots)}"
            steps.append(TraceStep(step_id, "update", "gradient-descent", "Move each parameter opposite its loss gradient.", r"\theta' = \theta-\eta\nabla_\theta L", f"w'={old_weight:g}-{learning_rate:g}({d_weight:g})={weight:g}; b'={old_bias:g}-{learning_rate:g}({d_bias:g})={bias:g}", ["parameters", "gradients", "learning_rate"], ["updated_parameters"], {"weight": [], "bias": [], "gradient_weight": [], "gradient_bias": []}, {"weight": [], "bias": []}, {**snapshot, "old_weight": old_weight, "old_bias": old_bias, "batch_x": xb, "batch_y": yb, "prediction": prediction}, code_mapping="weight -= learning_rate * gradient_weight"))
        if status != "completed":
            break
    document = trace_document("linear-regression-training", seed, {"learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "loss_reduction": "mean", "dataset": "y=2x+1"}, steps, initial, final_parameters={"weight": weight, "bias": bias}, warnings=[] if status == "completed" else ["Training diverged; the last finite state was preserved."])
    document.update({"dataset": {"x": x.reshape(-1).tolist(), "y": y.reshape(-1).tolist()}, "snapshots": snapshots, "status": status, "batch_mode": "full batch" if batch_size == len(x) else ("SGD" if batch_size == 1 else "mini-batch")})
    return document

