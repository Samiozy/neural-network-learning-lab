# Architecture decisions

## 2026-10-13 — Canonical Python numerical engine

FastAPI and NumPy own all authoritative calculation and trace creation. React owns
input gathering, presentation, formatting, and SVG geometry only. This prevents two
math implementations from drifting. A browser-only engine was rejected for that risk.

## 2026-10-13 — Column-vector convention and MSE definition

Hand calculations use column vectors and `W @ a + b`. MSE is mean squared error,
not half-squared error. These match the master plan defaults and are visible in traces.

## 2026-10-13 — Trace values inline for bounded MVP examples

Phase 0–6 tensors are capped at 16×16 and stored inline for transparent export and
replay. Reference-based snapshots are deferred until larger tensors make duplication
material. Trace schema 1.0.0 leaves that migration path explicit.

## 2026-10-13 — Deterministic linear training fixture

Phase 6 uses four fixed points from `y=2x+1`, seeded initialization, and batch order
without shuffling. This makes full-batch/mini-batch/SGD differences inspectable while
guaranteeing exact replay. XOR was rejected because it adds architecture complexity.

