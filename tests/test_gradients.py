"""Gradient check for the from-scratch MLP.

The whole point of this project is that backpropagation is implemented by
hand rather than delegated to a framework - which is only worth anything
if the hand-written gradients are actually correct. This test verifies
them the standard way: perturb one weight by +-eps, measure how the loss
really moves, and compare that finite difference against the gradient
backprop computed.

The analytic gradient is read back out of the real training code rather
than reimplemented here: one full-batch step updates W <- W - lr * dW, so

    dW = (W_before - W_after) / lr

which means this test checks the gradients that training actually uses.

Run:  python3 tests/test_gradients.py     (exits 1 on failure)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ml"))

from mlp import MLP  # noqa: E402

EPS = 1e-6        # perturbation for the finite difference
TOL = 1e-6        # max allowed relative error
N_CHECKS = 25     # weights sampled per configuration


def analytic_grads(model, X, y):
    """Gradients backprop actually applies, recovered from one full-batch step."""
    W0 = [W.copy() for W in model.W]
    b0 = [b.copy() for b in model.b]
    model.fit(X, y, epochs=1, batch_size=len(X), verbose=False)
    dW = [(w0 - w1) / model.lr for w0, w1 in zip(W0, model.W)]
    db = [(c0 - c1) / model.lr for c0, c1 in zip(b0, model.b)]
    # restore, so the caller keeps the pre-step model
    model.W = [w.copy() for w in W0]
    model.b = [c.copy() for c in b0]
    return dW, db


def numeric_grad(model, X, y, params, layer, index):
    """Central difference of the loss w.r.t. one parameter entry."""
    original = params[layer][index]

    params[layer][index] = original + EPS
    loss_plus = model.loss(X, y)

    params[layer][index] = original - EPS
    loss_minus = model.loss(X, y)

    params[layer][index] = original
    return (loss_plus - loss_minus) / (2 * EPS)


def check(name, hidden, fail_weight, l2, n_features=5, n_samples=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples)

    model = MLP(n_features=n_features, hidden=hidden,
                fail_weight=fail_weight, l2=l2, seed=seed)

    dW, db = analytic_grads(model, X, y)

    worst = 0.0
    for params, grads, tag in ((model.W, dW, "W"), (model.b, db, "b")):
        for _ in range(N_CHECKS):
            layer = rng.integers(0, len(params))
            index = tuple(rng.integers(0, s) for s in params[layer].shape)

            num = numeric_grad(model, X, y, params, layer, index)
            ana = grads[layer][index]

            denom = max(abs(num), abs(ana), 1e-12)
            rel = abs(num - ana) / denom
            worst = max(worst, rel)

            if rel > TOL:
                print(f"  FAIL {tag}[{layer}]{index}: "
                      f"numeric {num:+.8e} vs backprop {ana:+.8e} "
                      f"(rel err {rel:.2e})")
                return False, worst

    print(f"  ok   {name:<38} worst relative error {worst:.2e}")
    return True, worst


if __name__ == "__main__":
    print("gradient check (finite difference vs backprop)")
    print(f"eps = {EPS:g}, tolerance = {TOL:g}, "
          f"{N_CHECKS} weights + {N_CHECKS} biases per case\n")

    cases = [
        ("single hidden layer, unweighted", (16,), 1.0, 0.0),
        ("single hidden layer, fail_weight=5", (16,), 5.0, 0.0),
        ("two hidden layers", (32, 16), 5.0, 0.0),
        ("three hidden layers", (16, 8, 4), 5.0, 0.0),
        ("with L2 regularization", (16,), 5.0, 1e-3),
        ("no hidden layer (logistic)", (), 5.0, 0.0),
    ]

    results = [check(name, *cfg) for name, *cfg in
               [(n, h, fw, l2) for n, h, fw, l2 in cases]]

    print()
    if all(ok for ok, _ in results):
        print(f"PASS - {len(results)}/{len(results)} configurations")
        sys.exit(0)

    failed = sum(1 for ok, _ in results if not ok)
    print(f"FAIL - {failed}/{len(results)} configurations")
    sys.exit(1)
