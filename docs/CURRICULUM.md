# MVP curriculum

Every lab follows Concept → Formula → Worked example → Interactive trace →
Experiment → Check → Summary. The shared example moves from `[1,-2]` through a
2→3→1 tanh/identity network, MSE, all parameter gradients, then an update. The
single-neuron prerequisite uses `x=[2,1.5]`, `w=[.4,.7]`, `b=.1`, giving `z=1.95`.

1. Math: a six-part prerequisite path covering scalar arithmetic; vector/matrix rank
   and shape; transpose and explicit trailing-axis broadcasting; editable dot and
   matrix-vector products; derivative/finite-difference slope plots; and a chain-rule
   dependency path with prediction feedback and common misconceptions.
2. Neuron: weighted contribution, bias, identity/sigmoid/tanh/ReLU and saturation.
3. DNN: scalar neuron terms and full matrix form, shapes and parameter count.
4. Tasks/losses: continuous MSE, binary BCE with logits, multiclass softmax CE.
5. Backprop: local derivatives, deltas, outer-product parameter gradients, checks.
6. Training: full batch, mini-batch and SGD on deterministic `y=2x+1` data.

Limits: tiny examples illuminate mechanics but do not represent production scale;
loss need not decrease monotonically for arbitrary learning rates or batches; a low
training loss is not evidence of generalization. Validation/test topics begin in Phase 8.
