from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MathRequest(StrictModel):
    operation: Literal[
        "scalar", "add", "subtract", "multiply", "divide", "dot", "matvec",
        "transpose", "shape", "broadcast_add"
    ]
    left: Any
    right: Any | None = None


class DerivativeRequest(StrictModel):
    x: float = 0.75
    h: float = Field(1e-5, gt=0, le=0.1)


class NeuronRequest(StrictModel):
    x: list[float] = Field(default=[2.0, 1.5], min_length=1, max_length=16)
    w: list[float] = Field(default=[0.4, 0.7], min_length=1, max_length=16)
    b: float = 0.1
    activation: Literal["identity", "sigmoid", "tanh", "relu"] = "relu"
    seed: int = 7


class DenseLayer(StrictModel):
    weights: list[list[float]]
    bias: list[float]
    activation: Literal["identity", "sigmoid", "tanh", "relu"]

    @field_validator("weights")
    @classmethod
    def bounded_weights(cls, value: list[list[float]]) -> list[list[float]]:
        if not value or len(value) > 16 or any(not row or len(row) > 16 for row in value):
            raise ValueError("layers are limited to 16×16")
        return value


class DenseRequest(StrictModel):
    x: list[float] = Field(default=[1.0, -2.0], min_length=1, max_length=16)
    layers: list[DenseLayer]
    target: list[float] | None = None
    seed: int = 7


class LossRequest(StrictModel):
    kind: Literal["mse", "bce_logits", "softmax_ce"]
    prediction: list[float]
    target: list[float] | int


class TrainingRequest(StrictModel):
    seed: int = 7
    learning_rate: float = Field(0.1, gt=0, le=10)
    epochs: int = Field(10, ge=1, le=500)
    batch_size: int = Field(4, ge=1, le=4)
