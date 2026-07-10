import numpy as np

from dataset import load_train_test_by_run
from perceptron import standardize, evaluate


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Multi-layer perceptron: input -> hidden(ReLU) -> output(sigmoid).
#
# Why this breaks the single perceptron's ceiling: the good-wafer
# region is a BOX (spec_min <= x <= spec_max on each axis), and one
# linear boundary cannot carve out a box. But each hidden neuron is
# itself a little perceptron drawing one line - one can learn
# "Vth too low", another "Vth too high", another "Oxide too low", ...
# The output neuron then combines those verdicts (an AND-like vote),
# which is exactly a box. Stacking linear boundaries + nonlinearity
# is what buys the expressive power a single line cannot have.
class MLP:

    def __init__(self, n_features, n_hidden=16, learning_rate=0.1, seed=0):

        rng = np.random.default_rng(seed)

        # He initialization keeps ReLU activations at a healthy scale
        self.W1 = rng.normal(0, np.sqrt(2 / n_features), (n_features, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, 1))
        self.b2 = 0.0

        self.lr = learning_rate

    def forward(self, X):
        z1 = X @ self.W1 + self.b1     # hidden pre-activation
        a1 = relu(z1)                  # hidden output
        p = sigmoid(a1 @ self.W2 + self.b2).ravel()  # P(good)
        return z1, a1, p

    def predict_proba(self, X):
        return self.forward(X)[2]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    # Binary cross-entropy loss: heavily punishes confident mistakes.
    def loss(self, X, y):
        p = self.predict_proba(X)
        eps = 1e-12
        return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()

    # Mini-batch gradient descent with backpropagation.
    # Unlike the perceptron's "shift on mistake" rule, every sample
    # contributes a gradient telling each weight how to reduce the
    # loss, and errors at the output are propagated back through W2
    # to teach the hidden layer as well.
    def fit(self, X, y, epochs=30, batch_size=256):

        rng = np.random.default_rng(0)
        n = len(X)

        for epoch in range(1, epochs + 1):

            order = rng.permutation(n)

            for start in range(0, n, batch_size):

                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                m = len(xb)

                z1, a1, p = self.forward(xb)

                # output layer gradient (BCE + sigmoid simplifies to p - y)
                dz2 = (p - yb).reshape(-1, 1)
                dW2 = a1.T @ dz2 / m
                db2 = dz2.mean()

                # backpropagate through W2, gate by ReLU's derivative
                dz1 = (dz2 @ self.W2.T) * (z1 > 0)
                dW1 = xb.T @ dz1 / m
                db1 = dz1.mean(axis=0)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            if epoch % 3 == 0 or epoch == 1:
                acc = (self.predict(X) == y).mean()
                print(
                    f"epoch {epoch:2d} | "
                    f"loss {self.loss(X, y):.4f} | "
                    f"train accuracy {acc * 100:.2f}%"
                )


if __name__ == "__main__":

    # same run-based split as perceptron.py, so the two models are
    # compared on identical data
    X_train, y_train, X_test, y_test, test_runs = load_train_test_by_run()

    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    print(f"Train {len(X_train)} wafers / Test {len(X_test)} wafers")
    print(f"Held-out test runs : {', '.join(test_runs)}")
    print(f"Fail ratio (train) : {(y_train == 0).mean() * 100:.2f}%")
    print()

    model = MLP(n_features=X_train.shape[1])
    model.fit(X_train, y_train, epochs=30)

    print()
    print("=== Test set evaluation (cutoff 0.5) ===")
    evaluate(y_test, model.predict(X_test))

    # The MLP outputs P(good), so the operating point is a choice:
    # raising the cutoff catches more fails (recall) at the cost of
    # rejecting more good wafers (precision). A fab picks this point
    # from the cost of each error, not from accuracy.
    print()
    print("=== Threshold sweep - predict good when P(good) >= cutoff ===")
    proba = model.predict_proba(X_test)
    print(" cutoff | precision(fail) | recall(fail) |   F1")

    for cutoff in (0.1, 0.3, 0.5, 0.7, 0.9):

        pred = (proba >= cutoff).astype(int)

        tp = ((y_test == 0) & (pred == 0)).sum()
        fp = ((y_test == 1) & (pred == 0)).sum()
        fn = ((y_test == 0) & (pred == 1)).sum()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"   {cutoff:.1f}  |     {precision * 100:6.2f}%     |"
            f"   {recall * 100:6.2f}%    | {f1 * 100:6.2f}%"
        )
