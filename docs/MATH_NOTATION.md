# Mathematical notation

NNLL uses column vectors for hand calculations. At layer \(l\),
\(z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}\) and \(a^{(l)}=\phi(z^{(l)})\).
If a layer has \(n_l\) units and receives \(n_{l-1}\) values, then
\(W^{(l)}\in\mathbb R^{n_l\times n_{l-1}}\), while activations and biases are
\(n_l\times1\) columns. JSON flattens one-dimensional user inputs only at the API
boundary; the engine restores columns before matrix operations.

| Symbol | Meaning | Chosen or learned |
|---|---|---|
| \(x\) | input feature column | observed input |
| \(W,w\) | weight matrix/vector | learned parameter |
| \(b\) | bias | learned parameter |
| \(z\) | preactivation | intermediate |
| \(a,\hat y\) | activation/prediction | intermediate |
| \(y\) | target | observed training data |
| \(L\) | scalar loss | computed result |
| \(\eta\) | learning rate | user hyperparameter |
| \(\delta^{(l)}\) | \(\partial L/\partial z^{(l)}\) | gradient intermediate |

MSE always means \(N^{-1}\sum_i(\hat y_i-y_i)^2\), so its prediction gradient
is \(2(\hat y-y)/N\). The half-squared-error convention is not used by the MVP.
Batch examples place the batch axis first and average gradients over the batch.
Internal arithmetic is float64. The UI rounds to four decimals unless exact values
are expanded; comparisons use an absolute tolerance of `1e-6` unless stated.
At ReLU's nondifferentiable point zero, NNLL deliberately selects subgradient 0.

