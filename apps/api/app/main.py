from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.losses import binary_cross_entropy_from_logits, mse, softmax_cross_entropy
from app.core.tensors import derivative_trace, math_operation
from app.core.tracing import TraceStep, trace_document
from app.schemas.simulations import DenseRequest, DerivativeRequest, LossRequest, MathRequest, NeuronRequest, TrainingRequest
from app.simulators.dense import backpropagate, forward_dense, neuron_trace
from app.simulators.training import train_linear

app = FastAPI(title="Neural Network Learning Laboratory API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(ValueError)
async def value_error_handler(_request: object, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "trace_version": "1.0.0"}


@app.post("/api/math")
def calculate_math(request: MathRequest) -> dict[str, object]:
    return math_operation(request.operation, request.left, request.right)


@app.post("/api/derivative")
def calculate_derivative(request: DerivativeRequest) -> dict[str, object]:
    return derivative_trace(request.x, request.h)


@app.post("/api/neuron")
def calculate_neuron(request: NeuronRequest) -> dict[str, object]:
    return neuron_trace(request.x, request.w, request.b, request.activation, request.seed)


@app.post("/api/dense")
def calculate_dense(request: DenseRequest) -> dict[str, object]:
    layers = [layer.model_dump() for layer in request.layers]
    if request.target is not None:
        return backpropagate(request.x, request.target, layers, request.seed)
    return forward_dense(request.x, layers, request.seed)[0]


@app.post("/api/loss")
def calculate_loss(request: LossRequest) -> dict[str, object]:
    prediction = np.asarray(request.prediction, dtype=np.float64)
    if request.kind == "mse":
        if not isinstance(request.target, list):
            raise ValueError("MSE target must be a list")
        loss, gradient = mse(prediction, np.asarray(request.target, dtype=np.float64))
        values = {"loss": loss, "gradient": gradient, "prediction": prediction, "target": request.target}
        equation = r"L=\frac1N\sum_i(\hat y_i-y_i)^2"
    elif request.kind == "bce_logits":
        if not isinstance(request.target, list):
            raise ValueError("BCE target must be a list")
        loss, gradient = binary_cross_entropy_from_logits(prediction, np.asarray(request.target, dtype=np.float64))
        values = {"loss": loss, "gradient_logits": gradient, "logits": prediction, "target": request.target}
        equation = r"L=mean(max(z,0)-zy+\log(1+e^{-|z|}))"
    else:
        if not isinstance(request.target, int):
            raise ValueError("softmax CE target must be one integer class index")
        loss, gradient, probabilities = softmax_cross_entropy(prediction, request.target)
        values = {"loss": loss, "gradient_logits": gradient, "probabilities": probabilities, "logits": prediction, "target_index": request.target}
        equation = r"L=-z_y+\log\sum_j e^{z_j}"
    step = TraceStep("loss-1", "loss", request.kind, "Compute the selected task loss with a numerically stable formula.", equation, f"values={values}", ["prediction", "target"], ["loss"], {"prediction": list(prediction.shape)}, {"loss": []}, values)
    return trace_document("loss-lab", 0, {"kind": request.kind, "reduction": "mean"}, [step])


@app.post("/api/training")
def calculate_training(request: TrainingRequest) -> dict[str, object]:
    return train_linear(request.seed, request.learning_rate, request.epochs, request.batch_size)
