from __future__ import annotations

from typing import Any

import numpy as np

from .tracing import TraceStep, trace_document


def math_operation(operation: str, left: Any, right: Any | None = None) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = None if right is None else np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(a)) or (b is not None and not np.all(np.isfinite(b))):
        raise ValueError("inputs must contain only finite numbers")
    if operation == "dot":
        if b is None or a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            raise ValueError(f"dot requires equal one-dimensional shapes; received {a.shape} and {None if b is None else b.shape}")
        terms = a * b
        result: Any = float(np.sum(terms))
        equation = r"x \cdot y=\sum_i x_i y_i"
        substitution = "+".join(f"({x:g}\\times{y:g})" for x, y in zip(a, b, strict=True)) + f"={result:g}"
    elif operation == "matvec":
        if b is None or a.ndim != 2 or b.ndim != 1 or a.shape[1] != b.shape[0]:
            raise ValueError(f"matvec requires (m,n)·(n,); received {a.shape} and {None if b is None else b.shape}")
        result = a @ b
        terms = a * b
        equation = r"y_i=\sum_j A_{ij}x_j"
        substitution = ";\\;".join("+".join(f"({x:g}\\times{y:g})" for x, y in zip(row, b, strict=True)) for row in a)
    elif operation == "transpose":
        result = a.T
        terms = a
        equation = r"(A^T)_{ij}=A_{ji}"
        substitution = f"{a.tolist()}^T={result.tolist()}"
    elif operation == "shape":
        result = list(a.shape)
        terms = a
        equation = r"\operatorname{shape}(A)=(d_1,d_2,\ldots,d_r)"
        substitution = f"shape({a.tolist()})={result}"
    elif operation in {"scalar", "add", "subtract", "multiply", "divide"}:
        if b is None or a.size != 1 or b.size != 1:
            raise ValueError("scalar arithmetic requires two scalar values")
        left_value, right_value = float(a.item()), float(b.item())
        if operation in {"scalar", "add"}:
            result, symbol, equation = left_value + right_value, "+", r"c=a+b"
        elif operation == "subtract":
            result, symbol, equation = left_value - right_value, "-", r"c=a-b"
        elif operation == "multiply":
            result, symbol, equation = left_value * right_value, r"\times", r"c=ab"
        else:
            if right_value == 0:
                raise ValueError("division by zero is undefined")
            result, symbol, equation = left_value / right_value, r"\div", r"c=\frac{a}{b}"
        terms = np.array([a.item(), b.item()])
        substitution = f"{left_value:g}{symbol}{right_value:g}={result:g}"
    elif operation == "broadcast_add":
        if b is None:
            raise ValueError("broadcast addition requires two operands")
        try:
            result = a + b
        except ValueError as exc:
            raise ValueError(
                f"broadcasting compares dimensions from right to left; each pair must match or contain 1, received {a.shape} and {b.shape}"
            ) from exc
        terms = result
        equation = r"C_{ij}=A_{ij}+b_j"
        substitution = f"{a.tolist()}+{b.tolist()}={result.tolist()}"
    else:
        raise ValueError(f"unsupported operation: {operation}")
    result_array = np.asarray(result)
    step = TraceStep("math-1", "forward", operation, "Apply the selected operation.", equation, substitution,
                     ["left"] + ([] if b is None else ["right"]), ["result"],
                     {"left": list(a.shape), **({} if b is None else {"right": list(b.shape)})},
                     {"result": list(result_array.shape)}, {"left": a, "right": b, "terms": terms, "result": result},
                     code_mapping="result = left @ right" if operation == "matvec" else (
                         "result = left.shape" if operation == "shape" else f"result = {operation}(left, right)"
                     ))
    return trace_document("math-foundations", 0, {"operation": operation}, [step])


def derivative_trace(x: float, h: float = 1e-5) -> dict[str, Any]:
    if h <= 0:
        raise ValueError("h must be positive")
    # f(x)=tanh(x^2); f'=(1-tanh(x^2)^2)2x
    inner = x * x
    outer = float(np.tanh(inner))
    analytic = float((1.0 - outer * outer) * 2.0 * x)
    finite = float((np.tanh((x + h) ** 2) - np.tanh((x - h) ** 2)) / (2 * h))
    sample_count = 9
    sample_x = np.linspace(x - 1.0, x + 1.0, sample_count)
    curve = [{"x": float(point), "y": float(np.tanh(point**2))} for point in sample_x]
    tangent = [{"x": float(point), "y": float(outer + analytic * (point - x))} for point in sample_x]
    steps = [
        TraceStep("derivative-inner", "forward", "square", "Evaluate the inner function g(x)=x².", r"u=g(x)=x^2", f"u={x:g}^2={inner:g}", ["x"], ["u"], {"x": []}, {"u": []}, {"x": x, "u": inner}),
        TraceStep("derivative-finite-difference", "backward", "central-finite-difference", "Estimate the slope using nearby function values on both sides of x.", r"f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}", f"[tanh(({x:g}+{h:g})^2)-tanh(({x:g}-{h:g})^2)]/(2\\times{h:g})={finite:g}", ["x", "h"], ["finite_slope"], {"x": [], "h": []}, {"finite_slope": []}, {"x": x, "h": h, "f_x_plus_h": float(np.tanh((x+h)**2)), "f_x_minus_h": float(np.tanh((x-h)**2)), "finite_difference": finite}),
        TraceStep("derivative-chain", "backward", "chain-rule", "Multiply the outer and inner local derivatives because x changes u, which changes f.", r"\frac{df}{dx}=\frac{df}{du}\frac{du}{dx}", f"(1-\\tanh({inner:g})^2)(2\\times{x:g})={analytic:g}", ["x", "u"], ["df_dx"], {"x": [], "u": []}, {"df_dx": []}, {"outer_local_slope": 1.0-outer*outer, "inner_local_slope": 2*x, "analytic": analytic, "finite_difference": finite, "curve": curve, "tangent": tangent, "sample_count": sample_count}, derivation_notes=["Central difference independently estimates the local slope.", "Agreement within tolerance supports—but does not prove—the symbolic derivative."])
    ]
    return trace_document("chain-rule", 0, {"function": "tanh(x^2)", "h": h}, steps)
