# Phases 0–6 implementation report

## Goals and vertical slices

The MVP connects validated learner controls to pure NumPy computations, a versioned
trace, synchronized equation/value/shape presentation, exercises, and automated tests.
The API is canonical; the UI does not invent numerical output.

## Equations

- Dot product: `x·y=Σ xᵢyᵢ`; matrix-vector product: `yᵢ=Σ Aᵢⱼxⱼ`.
- Chain rule: `df/dx=(df/du)(du/dx)` with a central-difference check.
- Neuron/layer: `z=wᵀx+b`; `z⁽ˡ⁾=W⁽ˡ⁾a⁽ˡ⁻¹⁾+b⁽ˡ⁾`; `a=φ(z)`.
- MSE, stable BCE-with-logits, and log-sum-exp softmax cross-entropy.
- Backprop: `δ⁽ˡ⁾=(∂L/∂a⁽ˡ⁾)⊙φ′(z⁽ˡ⁾)`, `∂L/∂W=δaᵀ`, `∂L/∂b=δ`.
- Gradient descent: `θ′=θ−η∇θL`.

## UI interactions

Six keyboard-operable tabs expose preset calculations, editable neuron values and
activation, stored-step navigation, exact JSON/code disclosure, loss comparisons,
gradient prediction, training controls, and a synchronized loss curve. Static math and
tables remain available without animation; reduced-motion preferences are honored.

## Test strategy and dependencies

Python tests independently reconstruct fixtures and finite differences, exercise
stability/edge cases, validate API errors, and test deterministic replay. Vitest checks
stored forward/back navigation. Next build/type/lint checks the UI. Dependencies are
pinned in `pyproject.toml` and both package manifests. Phases build sequentially on the
trace and math conventions established in Phase 0.

## Verification evidence — 2026-10-13

- `./.venv/bin/pytest -q`: 18 passed (one upstream Starlette deprecation warning).
- `./.venv/bin/ruff check apps/api`: passed.
- `./.venv/bin/mypy apps/api/app`: passed under strict configuration.
- `npm --prefix apps/web test`: 2 component workflows passed.
- `npm --prefix apps/web run typecheck`, `lint`, and `build`: passed.
- `npm --prefix apps/web audit --omit=dev`: zero runtime vulnerabilities.

The remaining npm report is confined to `eslint-config-next`'s globbing toolchain;
npm offers only an incompatible downgrade. It is recorded rather than concealed.
