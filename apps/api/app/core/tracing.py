from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np

Phase = Literal["forward", "loss", "backward", "update"]


def json_value(value: Any) -> Any:
    """Convert numerical state to strict, JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("Trace values must be finite")
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Trace values must be finite")
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


@dataclass
class TraceStep:
    step_id: str
    phase: Phase
    operation: str
    description: str
    equation_latex: str
    substitution_latex: str
    input_refs: list[str]
    output_refs: list[str]
    input_shapes: dict[str, list[int]]
    output_shapes: dict[str, list[int]]
    values: dict[str, Any]
    tensor_indices: list[list[int]] = field(default_factory=list)
    highlight_targets: list[str] = field(default_factory=list)
    derivation_notes: list[str] = field(default_factory=list)
    code_mapping: str = ""

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json_value(asdict(self)))


def trace_document(
    architecture: str,
    seed: int,
    config: dict[str, Any],
    steps: list[TraceStep],
    initial_parameters: dict[str, Any] | None = None,
    gradients: dict[str, Any] | None = None,
    final_parameters: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return cast(dict[str, Any], json_value({
        "trace_version": "1.0.0",
        "simulation_id": str(uuid4()),
        "architecture": architecture,
        "seed": seed,
        "precision_policy": {"internal": "float64", "display_decimals": 4, "atol": 1e-6},
        "config": config,
        "steps": [step.to_dict() for step in steps],
        "initial_parameters": initial_parameters or {},
        "gradients": gradients or {},
        "final_parameters": final_parameters or initial_parameters or {},
        "warnings": warnings or [],
        "errors": [],
    }))
