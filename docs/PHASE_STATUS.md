# Phase status

Verification evidence is updated only after commands actually run.

| Phase | Status | Features delivered | Worked example | Tests executed | Known limitations | Next action |
|---|---|---|---|---|---|---|
| 0 | verified | Monorepo, app shell, health API, trace 1.0, strict configs, CI/docs | Typed neuron trace prototype | pytest/API smoke; Vitest; Ruff; mypy; ESLint; tsc; Next build | Python 3.11+ required; 5 dev-only lint-chain advisories with no nonbreaking upstream fix; runtime audit clean | Await Phase 7 request |
| 1 | verified | Six-part prerequisite path; scalar arithmetic, rank/shape, transpose, broadcasting, editable dot/matvec, slope plot, finite differences, chain-rule dependency view, quiz feedback | `[2,3]·[4,5]=23`, `[[1,2],[3,4]]·[2,1]=[4,10]`, `tanh(x²)` | Core arithmetic/shape/broadcast tests, chain-rule numerical check, UI path test | Broadcasting teaches NumPy's trailing-axis convention on small bounded tensors | Await Phase 7 request |
| 2 | verified | Editable neuron, four activations, contribution ledger, preset reset | `z=1.95`, ReLU | neuron fixture, extreme sigmoid, nonfinite tests | Two-input UI; API accepts up to 16 | Await Phase 7 request |
| 3 | verified | Canonical 2→3→1 forward engine, scalar terms, matrix trace | Fixed tanh/identity network | independent dense fixture and parameter-count test | UI uses fixed layer parameters | Await Phase 7 request |
| 4 | verified | MSE, stable BCE logits, stable softmax CE, explicit reductions | Regression/binary/3-class fixtures | stability and probability-sum tests | One-sample task UI | Await Phase 7 request |
| 5 | verified | Full forward/loss/backward ledger, every W/b gradient, ReLU convention | Same 2→3→1 fixture | all 13 parameter gradients vs central differences (`1e-6`) | PyTorch parity optional/not installed | Await Phase 7 request |
| 6 | verified | Seeded real train loop, batch modes, loss chart, safe divergence | Four-point `y=2x+1` | exact replay, batch-mode, loss-decrease tests | No shuffling or decision boundary for 1D data | Await Phase 7 request |
