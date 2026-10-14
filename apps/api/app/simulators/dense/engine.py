from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.core.activations import activate, activation_derivative
from app.core.losses import mse
from app.core.tracing import TraceStep, trace_document


Array = NDArray[np.float64]


def _array(value: Any, ndim: int, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D tensor; received shape {array.shape}")
    return array


def neuron_trace(x: Any, w: Any, b: float, activation: str = "relu", seed: int = 7) -> dict[str, Any]:
    inputs, weights = _array(x, 1, "x"), _array(w, 1, "w")
    if inputs.shape != weights.shape:
        raise ValueError(f"x and w shapes must match; received {inputs.shape} and {weights.shape}")
    products = inputs * weights
    z = float(np.sum(products) + b)
    a = float(activate(np.array([z]), activation)[0])
    multiplication = "+".join(f"({wi:g}\\times{xi:g})" for wi, xi in zip(weights, inputs, strict=True))
    steps = [
        TraceStep("neuron-weighted-sum", "forward", "weighted-sum", "Each weight scales its input; the bias shifts the sum.", r"z=w^Tx+b", f"z={multiplication}+{b:g}={z:g}", ["x", "w", "b"], ["z"], {"x": [inputs.size, 1], "w": [1, inputs.size], "b": [1]}, {"z": [1]}, {"x": inputs, "w": weights, "b": b, "products": products, "partial_sums": np.cumsum(products), "z": z}, highlight_targets=[f"weight-{i}" for i in range(inputs.size)], code_mapping="z = w @ x + b"),
        TraceStep("neuron-activation", "forward", activation, f"The {activation} activation transforms the preactivation.", r"a=\phi(z)", f"a={activation}({z:g})={a:g}", ["z"], ["a"], {"z": [1]}, {"a": [1]}, {"z": z, "a": a}, highlight_targets=["activation"], code_mapping=f"a = {activation}(z)"),
    ]
    return trace_document("single-neuron", seed, {"activation": activation}, steps, {"w": weights, "b": b})


def forward_dense(x: Any, layers: list[dict[str, Any]], seed: int = 7) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    activation: Array = _array(x, 1, "x").reshape(-1, 1)
    initial: dict[str, Any] = {"layers": []}
    cache: list[dict[str, Any]] = []
    steps: list[TraceStep] = []
    for index, layer in enumerate(layers, start=1):
        weights = _array(layer["weights"], 2, f"W{index}")
        bias = _array(layer["bias"], 1, f"b{index}").reshape(-1, 1)
        name = str(layer.get("activation", "identity"))
        if weights.shape[1] != activation.shape[0] or weights.shape[0] != bias.shape[0]:
            raise ValueError(f"layer {index}: W{weights.shape}, input{activation.shape}, and b{bias.shape} are incompatible")
        previous = activation
        z: Array = np.asarray(weights @ previous + bias, dtype=np.float64)
        activation = np.asarray(activate(z, name), dtype=np.float64)
        cache.append({"a_previous": previous, "W": weights, "b": bias, "z": z, "a": activation, "activation_name": np.array(name)})
        initial["layers"].append({"weights": weights, "bias": bias.reshape(-1), "activation": name})
        neuron_terms = [[float(weights[i, j] * previous[j, 0]) for j in range(weights.shape[1])] for i in range(weights.shape[0])]
        steps.extend([
            TraceStep(f"layer-{index}-linear", "forward", "matrix-vector-product", f"Layer {index} combines every incoming activation with learned weights and a bias.", rf"z^{{({index})}}=W^{{({index})}}a^{{({index-1})}}+b^{{({index})}}", f"z={weights.tolist()}@{previous.reshape(-1).tolist()}+{bias.reshape(-1).tolist()}={z.reshape(-1).tolist()}", [f"W{index}", f"a{index-1}", f"b{index}"], [f"z{index}"], {f"W{index}": list(weights.shape), f"a{index-1}": list(previous.shape), f"b{index}": list(bias.shape)}, {f"z{index}": list(z.shape)}, {"W": weights, "a_previous": previous, "b": bias, "neuron_terms": neuron_terms, "z": z}, highlight_targets=[f"layer-{index}", *[f"neuron-{index}-{i}" for i in range(weights.shape[0])]], code_mapping=f"z{index} = W{index} @ a{index-1} + b{index}"),
            TraceStep(f"layer-{index}-activation", "forward", name, f"Apply {name} elementwise at layer {index}.", rf"a^{{({index})}}=\phi(z^{{({index})}})", f"a={name}({z.reshape(-1).tolist()})={activation.reshape(-1).tolist()}", [f"z{index}"], [f"a{index}"], {f"z{index}": list(z.shape)}, {f"a{index}": list(activation.shape)}, {"z": z, "a": activation}, code_mapping=f"a{index} = {name}(z{index})"),
        ])
    document = trace_document("dense-network", seed, {"layer_sizes": [cache[0]["a_previous"].shape[0], *[c["a"].shape[0] for c in cache]]}, steps, initial)
    document["output"] = activation.reshape(-1).tolist()
    document["parameter_count"] = int(sum(c["W"].size + c["b"].size for c in cache))
    return document, cache


def backpropagate(x: Any, target: Any, layers: list[dict[str, Any]], seed: int = 7) -> dict[str, Any]:
    document, cache = forward_dense(x, layers, seed)
    prediction = cache[-1]["a"]
    target_array = _array(target, 1, "target").reshape(-1, 1)
    loss, gradient = mse(prediction, target_array)
    document["steps"].append(TraceStep("loss-mse", "loss", "mean-squared-error", "MSE averages squared prediction errors.", r"L=\frac1N\sum_i(\hat y_i-y_i)^2", f"L=mean(({prediction.reshape(-1).tolist()}-{target_array.reshape(-1).tolist()})^2)={loss:g}", [f"a{len(cache)}", "target"], ["loss"], {f"a{len(cache)}": list(prediction.shape), "target": list(target_array.shape)}, {"loss": []}, {"prediction": prediction, "target": target_array, "loss": loss, "dL_da": gradient}, code_mapping="loss = mean((prediction - target) ** 2)").to_dict())
    gradients: dict[str, Any] = {}
    for zero_index in reversed(range(len(cache))):
        layer_index = zero_index + 1
        item = cache[zero_index]
        local = activation_derivative(item["z"], str(item["activation_name"].item()))
        upstream = gradient.copy()
        delta = upstream * local
        d_w = delta @ item["a_previous"].T
        d_b = delta.copy()
        gradient = item["W"].T @ delta
        gradients[f"W{layer_index}"] = d_w
        gradients[f"b{layer_index}"] = d_b.reshape(-1)
        document["steps"].extend([
            TraceStep(f"backward-{layer_index}-delta", "backward", "chain-rule", f"Layer {layer_index} combines the upstream gradient with its activation slope.", rf"\delta^{{({layer_index})}}=\frac{{\partial L}}{{\partial a^{{({layer_index})}}}}\odot\phi'(z^{{({layer_index})}})", f"delta={upstream.reshape(-1).tolist()}*{local.reshape(-1).tolist()}={delta.reshape(-1).tolist()}", [f"dL_da{layer_index}", f"z{layer_index}"], [f"delta{layer_index}"], {f"dL_da{layer_index}": list(upstream.shape), f"z{layer_index}": list(item["z"].shape)}, {f"delta{layer_index}": list(delta.shape)}, {"upstream": upstream, "activation_derivative": local, "delta": delta}, derivation_notes=["ReLU uses subgradient 0 exactly at z=0."], code_mapping="delta = upstream * activation_derivative(z)"),
            TraceStep(f"backward-{layer_index}-parameters", "backward", "parameter-gradients", "Outer products assign a derivative to every weight; biases receive delta directly.", rf"\frac{{\partial L}}{{\partial W^{{({layer_index})}}}}=\delta^{{({layer_index})}}(a^{{({layer_index-1})}})^T", f"dW={delta.tolist()}@{item['a_previous'].T.tolist()}={d_w.tolist()}", [f"delta{layer_index}", f"a{layer_index-1}"], [f"dW{layer_index}", f"db{layer_index}"], {f"delta{layer_index}": list(delta.shape), f"a{layer_index-1}": list(item["a_previous"].shape)}, {f"dW{layer_index}": list(d_w.shape), f"db{layer_index}": list(d_b.shape)}, {"dW": d_w, "db": d_b, "dL_da_previous": gradient}, code_mapping="dW = delta @ a_previous.T; db = delta"),
        ])
    document["gradients"] = {key: value.tolist() for key, value in gradients.items()}
    document["loss"] = loss
    document["target"] = target_array.reshape(-1).tolist()
    return document
