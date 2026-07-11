import argparse
import os

import numpy as np

from dataset import load_train_test_by_run
from perceptron import standardize, evaluate
from metrics import binary_metrics

MODEL_PATH = "graph/ml/mlp_model.npz"


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Multi-layer perceptron: input -> hidden(ReLU) x N -> output(sigmoid).
#
# Why this breaks the single perceptron's ceiling: the good-wafer
# region is a BOX (spec_min <= x <= spec_max on each axis), and one
# linear boundary cannot carve out a box. But each hidden neuron is
# itself a little perceptron drawing one line - one can learn
# "Vth too low", another "Vth too high", another "Oxide too low", ...
# The output neuron then combines those verdicts (an AND-like vote),
# which is exactly a box. Stacking linear boundaries + nonlinearity
# is what buys the expressive power a single line cannot have. Extra
# hidden layers let those pieces be composed further.
class MLP:

    # hidden : int or sequence of layer widths, e.g. 16 or (32, 16).
    # fail_weight scales how much a fail(0) sample counts in the loss.
    #   In a fab a shipped bad wafer (false negative) costs far more than
    #   a scrapped good one (false positive), yet fails are the minority,
    #   so an unweighted model drifts toward "call everything good".
    #   fail_weight > 1 bakes that cost asymmetry into training itself,
    #   rather than only patching it afterwards by moving the threshold.
    # l2 : optional weight decay to curb overfitting as the net grows.
    def __init__(self, n_features, hidden=(16,), learning_rate=0.1,
                 fail_weight=1.0, l2=0.0, seed=0):

        if isinstance(hidden, (int, np.integer)):
            hidden = (int(hidden),)
        self.hidden = tuple(int(h) for h in hidden)
        self.lr = learning_rate
        self.fail_weight = fail_weight
        self.l2 = l2
        self.seed = seed

        rng = np.random.default_rng(seed)
        sizes = [n_features] + list(self.hidden) + [1]

        # He initialization keeps ReLU activations at a healthy scale
        self.W = [
            rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
            for i in range(len(sizes) - 1)
        ]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    # per-sample loss weight: fail_weight for fails, 1 for goods
    def _weights(self, y):
        return np.where(y == 0, self.fail_weight, 1.0)

    # forward pass keeping every layer's pre-activation (z) and output
    # (a), which backprop needs. Hidden layers use ReLU, the last sigmoid.
    def _forward(self, X):
        zs = []
        acts = [X]
        a = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            zs.append(z)
            a = sigmoid(z) if i == last else relu(z)
            acts.append(a)
        return zs, acts, acts[-1].ravel()

    def predict_proba(self, X):
        return self._forward(X)[2]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    # Persist the trained model. Weights of every layer plus mean/std
    # (from standardization) are saved, because a new wafer must be
    # scaled the exact same way the training data was, or the model
    # sees garbage.
    def save(self, path, mean, std):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        layers = {f"W{i}": W for i, W in enumerate(self.W)}
        layers.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(
            path,
            n_layers=np.array(len(self.W)),
            hidden=np.array(self.hidden),
            fail_weight=np.array(self.fail_weight),
            l2=np.array(self.l2),
            mean=mean, std=std,
            **layers
        )

    # Rebuild a model from a saved file. Returns (model, mean, std).
    @classmethod
    def load(cls, path):
        data = np.load(path)
        n_layers = int(data["n_layers"])
        model = cls(
            n_features=data["W0"].shape[0],
            hidden=tuple(int(h) for h in data["hidden"]),
            fail_weight=float(data["fail_weight"]),
            l2=float(data["l2"]),
        )
        model.W = [data[f"W{i}"] for i in range(n_layers)]
        model.b = [data[f"b{i}"] for i in range(n_layers)]
        return model, data["mean"], data["std"]

    # Class-weighted binary cross-entropy (plus optional L2). Heavily
    # punishes confident mistakes, and counts fail-class mistakes
    # fail_weight times more.
    def loss(self, X, y):
        p = self.predict_proba(X)
        eps = 1e-12
        w = self._weights(y)
        data = -(w * (y * np.log(p + eps)
                      + (1 - y) * np.log(1 - p + eps))).mean()
        reg = 0.5 * self.l2 * sum((W ** 2).sum() for W in self.W)
        return data + reg

    # Mini-batch gradient descent with backpropagation.
    # Unlike the perceptron's "shift on mistake" rule, every sample
    # contributes a gradient telling each weight how to reduce the
    # loss, and the output error is propagated back through every
    # hidden layer to teach them all.
    #
    # Pass X_val/y_val to also track held-out loss/accuracy each epoch;
    # the returned history makes the train-vs-val gap (overfitting)
    # plottable.
    def fit(self, X, y, epochs=30, batch_size=256,
            X_val=None, y_val=None, verbose=True):

        rng = np.random.default_rng(self.seed)
        n = len(X)
        last = len(self.W) - 1
        history = []

        for epoch in range(1, epochs + 1):

            order = rng.permutation(n)

            for start in range(0, n, batch_size):

                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                m = len(xb)

                zs, acts, p = self._forward(xb)

                # output layer delta (BCE + sigmoid simplifies to p - y),
                # scaled per-sample by the class weight
                delta = (self._weights(yb) * (p - yb)).reshape(-1, 1)

                # walk backward, computing each layer's gradient and
                # propagating delta through its weights (ReLU-gated)
                for i in range(last, -1, -1):
                    dW = acts[i].T @ delta / m + self.l2 * self.W[i]
                    db = delta.mean(axis=0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * (zs[i - 1] > 0)
                    self.W[i] -= self.lr * dW
                    self.b[i] -= self.lr * db

            rec = {
                "epoch": epoch,
                "train_loss": self.loss(X, y),
                "train_acc": (self.predict(X) == y).mean(),
            }
            if X_val is not None:
                rec["val_loss"] = self.loss(X_val, y_val)
                rec["val_acc"] = (self.predict(X_val) == y_val).mean()
            history.append(rec)

            if verbose and (epoch % 3 == 0 or epoch == 1):
                msg = (f"epoch {epoch:2d} | loss {rec['train_loss']:.4f} | "
                       f"train acc {rec['train_acc'] * 100:.2f}%")
                if X_val is not None:
                    msg += f" | val acc {rec['val_acc'] * 100:.2f}%"
                print(msg)

        return history


def _train_one(name, X_train, y_train, X_test, y_test, args, fail_weight):
    print(f"### {name} (fail_weight = {fail_weight:g}, "
          f"hidden = {tuple(args.hidden)}, l2 = {args.l2:g}) ###")
    model = MLP(
        n_features=X_train.shape[1],
        hidden=args.hidden,
        learning_rate=args.lr,
        fail_weight=fail_weight,
        l2=args.l2,
        seed=args.seed,
    )
    model.fit(X_train, y_train, epochs=args.epochs,
              batch_size=args.batch_size, X_val=X_test, y_val=y_test)
    print()
    print("--- Test set evaluation (cutoff 0.5) ---")
    evaluate(y_test, model.predict(X_test))
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the from-scratch MLP wafer classifier"
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[16],
                        help="hidden layer widths, e.g. --hidden 32 16 "
                             "(default: 16)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fail-weight", type=float, default=5.0,
                        help="loss weight on fails for the saved model "
                             "(default: 5)")
    parser.add_argument("--l2", type=float, default=0.0,
                        help="L2 weight decay (default: 0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip the unweighted baseline comparison")
    args = parser.parse_args()

    # same run-based split as perceptron.py, so the two models are
    # compared on identical data
    X_train, y_train, X_test, y_test, test_runs = load_train_test_by_run()

    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    print(f"Train {len(X_train)} wafers / Test {len(X_test)} wafers")
    print(f"Held-out test runs : {', '.join(test_runs)}")
    print(f"Fail ratio (train) : {(y_train == 0).mean() * 100:.2f}%")
    print()

    # Baseline: unweighted loss (every wafer counts equally). Expect a
    # model that scores high accuracy but misses many fails.
    if not args.no_baseline:
        _train_one("Unweighted MLP", X_train, y_train, X_test, y_test,
                   args, fail_weight=1.0)
        print()

    # Weighted loss: a missed fail costs fail_weight x a rejected good,
    # applied during training. Expect higher fail recall at the same cutoff.
    model = _train_one("Weighted MLP", X_train, y_train, X_test, y_test,
                       args, fail_weight=args.fail_weight)

    # The MLP outputs P(good), so the operating point is also tunable
    # after training: raising the cutoff catches more fails (recall) at
    # the cost of rejecting more good wafers (precision). A fab picks
    # this point from the cost of each error, not from accuracy.
    print()
    print("=== Threshold sweep (weighted model) ===")
    proba = model.predict_proba(X_test)
    print(" cutoff | precision(fail) | recall(fail) |   F1")

    for cutoff in (0.1, 0.3, 0.5, 0.7, 0.9):
        m = binary_metrics(y_test, (proba >= cutoff).astype(int))
        print(f"   {cutoff:.1f}  |     {m['precision'] * 100:6.2f}%     |"
              f"   {m['recall'] * 100:6.2f}%    | {m['f1'] * 100:6.2f}%")

    # save the weighted model so ml/judge.py can grade fresh runs
    model.save(MODEL_PATH, mean, std)
    print()
    print(f"saved model -> {MODEL_PATH}")
