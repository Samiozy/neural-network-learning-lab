import json

import numpy as np
import pytest

from app.core.activations import activate, activation_derivative
from app.core.losses import binary_cross_entropy_from_logits, mse, softmax_cross_entropy
from app.core.tensors import derivative_trace, math_operation
from app.simulators.dense import backpropagate, forward_dense, neuron_trace
from app.simulators.training import train_linear


LAYERS = [
    {"weights": [[0.2, -0.4], [0.7, 0.1], [-0.5, 0.8]], "bias": [0.1, -0.2, 0.3], "activation": "tanh"},
    {"weights": [[0.6, -0.3, 0.5]], "bias": [0.2], "activation": "identity"},
]


def test_dot_worked_example_and_json_trace():
    trace = math_operation("dot", [2, 3], [4, 5])
    assert trace["steps"][0]["values"]["result"] == 23
    assert r"\times" in trace["steps"][0]["substitution_latex"]
    json.dumps(trace, allow_nan=False)


def test_shape_mismatch_explains_problem():
    with pytest.raises(ValueError, match="equal one-dimensional shapes"):
        math_operation("dot", [1, 2], [3])


def test_matrix_vector_and_transpose_shapes():
    assert math_operation("matvec", [[1, 2], [3, 4]], [2, 1])["steps"][0]["values"]["result"] == [4, 10]
    assert math_operation("transpose", [[1, 2, 3], [4, 5, 6]])["steps"][0]["output_shapes"]["result"] == [3, 2]


def test_scalar_arithmetic_shape_and_division_validation():
    assert math_operation("multiply", 6, 2)["steps"][0]["values"]["result"] == 12
    assert math_operation("shape", [[1, 2, 3], [4, 5, 6]])["steps"][0]["values"]["result"] == [2, 3]
    with pytest.raises(ValueError, match="division by zero"):
        math_operation("divide", 2, 0)


def test_broadcasting_convention_and_mismatch_explanation():
    trace = math_operation("broadcast_add", [[1, 2, 3], [4, 5, 6]], [10, 20, 30])
    assert trace["steps"][0]["values"]["result"] == [[11, 22, 33], [14, 25, 36]]
    with pytest.raises(ValueError, match="right to left"):
        math_operation("broadcast_add", [[1, 2, 3], [4, 5, 6]], [10, 20])


def test_chain_rule_matches_central_difference():
    values = derivative_trace(0.75)["steps"][-1]["values"]
    assert values["analytic"] == pytest.approx(values["finite_difference"], abs=1e-8)
    assert len(values["curve"]) == 9


def test_stable_activations_and_relu_zero_subgradient():
    values = activate(np.array([-1000.0, 0.0, 1000.0]), "sigmoid")
    assert np.all(np.isfinite(values)) and values.tolist() == pytest.approx([0, 0.5, 1])
    assert activation_derivative(np.array([0.0]), "relu")[0] == 0


def test_single_neuron_reference():
    trace = neuron_trace([2, 1.5], [0.4, 0.7], 0.1, "relu")
    assert trace["steps"][0]["values"]["products"] == pytest.approx([0.8, 1.05])
    assert trace["steps"][1]["values"]["a"] == pytest.approx(1.95)


def test_dense_forward_fixture():
    trace, _ = forward_dense([1, -2], LAYERS)
    expected_hidden = np.tanh(np.array([1.1, 0.3, -1.8]))
    expected = np.array([[0.6, -0.3, 0.5]]) @ expected_hidden + 0.2
    assert trace["output"] == pytest.approx(expected.tolist())
    assert trace["parameter_count"] == 13


@pytest.mark.parametrize("kind", ["mse", "bce", "ce"])
def test_losses_are_finite(kind):
    if kind == "mse":
        value, grad = mse(np.array([1.0]), np.array([2.0]))
    elif kind == "bce":
        value, grad = binary_cross_entropy_from_logits(
            np.array([1000.0, -1000.0]), np.array([1.0, 0.0])
        )
    else:
        value, grad, probabilities = softmax_cross_entropy(
            np.array([1000.0, 999.0, -1000.0]), 0
        )
        assert probabilities.sum() == pytest.approx(1)
    assert np.isfinite(value) and np.all(np.isfinite(grad))


def test_backprop_all_parameters_match_central_differences():
    analytical = backpropagate([1, -2], [0.5], LAYERS)["gradients"]
    epsilon = 1e-6
    for layer_index, layer in enumerate(LAYERS, start=1):
        for field, gradient_key in (("weights", f"W{layer_index}"), ("bias", f"b{layer_index}")):
            array = np.asarray(layer[field], dtype=float)
            for index in np.ndindex(array.shape):
                plus = [{**item, "weights": np.array(item["weights"], float).copy(), "bias": np.array(item["bias"], float).copy()} for item in LAYERS]
                minus = [{**item, "weights": np.array(item["weights"], float).copy(), "bias": np.array(item["bias"], float).copy()} for item in LAYERS]
                plus[layer_index - 1][field][index] += epsilon
                minus[layer_index - 1][field][index] -= epsilon
                loss_plus = backpropagate([1, -2], [0.5], plus)["loss"]
                loss_minus = backpropagate([1, -2], [0.5], minus)["loss"]
                numerical = (loss_plus - loss_minus) / (2 * epsilon)
                assert np.asarray(analytical[gradient_key])[index] == pytest.approx(numerical, abs=1e-6)


def test_training_replay_loss_decrease_and_batch_modes():
    first = train_linear(seed=7, learning_rate=0.1, epochs=20, batch_size=4)
    second = train_linear(seed=7, learning_rate=0.1, epochs=20, batch_size=4)
    assert first["snapshots"] == second["snapshots"]
    assert first["snapshots"][-1]["loss"] < first["snapshots"][0]["loss"]
    assert train_linear(batch_size=1)["batch_mode"] == "SGD"


def test_invalid_nonfinite_input_rejected():
    with pytest.raises(ValueError, match="finite"):
        neuron_trace([np.inf], [1], 0)
