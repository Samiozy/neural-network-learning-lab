# Neural Network Learning Laboratory

![Neural Network Learning Lab infographic showing a deep neural network, convolutional neural network, recurrent neural network, and their core equations](docs/assets/neural-network-learning-lab-infographic.png)

NNLL is a local-first, inspectable neural-network microscope. The Phase 0–6 MVP lets
you follow arithmetic through a neuron and 2→3→1 DNN, stable losses, every parameter
gradient, and genuine deterministic gradient-descent updates.

## Requirements

- macOS, Node.js 20+, npm 10+
- Python 3.11+ (`brew install python@3.11` if needed)

## Fresh-clone setup

```bash
python3.11 -m venv .venv
./.venv/bin/pip install -e '.[dev]'
npm install
npm --prefix apps/web install
```

Run both services:

```bash
npm run dev
```

Open http://localhost:3000. The API and health response are at
http://localhost:8000 and http://localhost:8000/health. Once dependencies are installed,
the product runtime makes no external requests.

## Checks

```bash
npm test
npm run lint
npm run typecheck
npm --prefix apps/web run build
```

## Learning path

Use tabs in order. In every lab, make a prediction, run the calculation, then use the
ledger's step controls. Expand exact values only after inspecting the symbolic equation
and numeric substitution. Resetting a ledger replay does not reset learned parameters;
starting training again with the same seed reconstructs the entire run.

Canonical requirements and conventions live in `docs/MASTER_PLAN.md`,
`docs/MATH_NOTATION.md`, and `docs/TRACE_SCHEMA.md`.
