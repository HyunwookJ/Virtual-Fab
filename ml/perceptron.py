import numpy as np

from dataset import load_train_test_by_run, FEATURE_COLUMNS
from metrics import print_evaluation as evaluate


# Features have very different scales (Vth ~0.7, Oxide ~100, Leakage ~1-10),
# so standardize each to mean 0 / std 1 before training.
# Test data must reuse the training set's mean/std.
def standardize(X, mean=None, std=None):

    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)

    return (X - mean) / std, mean, std


# Classic single-layer perceptron, implemented from scratch.
#
# Decision rule : predict good(1) if  w . x + b >= 0  else fail(0)
# Learning rule : whenever a wafer is misclassified,
#                   w <- w + lr * (y - y_pred) * x
#                   b <- b + lr * (y - y_pred)
# i.e. the linear boundary shifts toward every mistake it makes.
class Perceptron:

    def __init__(self, n_features, learning_rate=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.lr = learning_rate

    # signed distance to the boundary; higher = more "good". Used as a
    # ranking score for ROC/PR (the perceptron has no probability).
    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def fit(self, X, y, epochs=10):

        rng = np.random.default_rng(0)

        # --- Pocket algorithm ---
        # On data that is NOT linearly separable, the perceptron never
        # converges: every epoch it keeps hitting misclassified wafers,
        # so the boundary keeps moving forever. That means the weights
        # at the moment training happens to stop are arbitrary - if the
        # last epoch ended in a bad oscillation, we ship a bad model.
        #
        # The pocket algorithm fixes this: while the normal perceptron
        # updates run as usual, we keep a separate copy ("pocket") of
        # the best weights seen so far. Whenever the current weights
        # score a new best accuracy, we put them in the pocket. After
        # training, the pocket - not the final weights - becomes the
        # model. The result is "the best linear boundary found during
        # training" instead of "wherever the boundary happened to be
        # when we stopped".
        # (Classic pocket checks after every single update; we check
        # once per epoch, which is far cheaper and works just as well
        # in practice.)
        best_w = self.w.copy()
        best_b = self.b
        best_acc = (self.predict(X) == y).mean()

        for epoch in range(1, epochs + 1):

            # reshuffle every epoch so the boundary is not dragged
            # along the same trajectory of mistakes each time
            order = rng.permutation(len(X))

            updates = 0

            for i in order:

                xi, yi = X[i], y[i]

                pred = 1 if xi @ self.w + self.b >= 0 else 0
                error = yi - pred

                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    updates += 1

            acc = (self.predict(X) == y).mean()

            if acc > best_acc:
                best_w = self.w.copy()
                best_b = self.b
                best_acc = acc

            print(
                f"epoch {epoch:2d} | "
                f"updates {updates:5d} | "
                f"train accuracy {acc * 100:.2f}% | "
                f"pocket best {best_acc * 100:.2f}%"
            )

        # ship the pocket, not the last oscillation
        self.w = best_w
        self.b = best_b


# Accuracy alone is misleading here (predicting "all good" already
# scores high on this imbalanced data), so evaluation also reports the
# confusion matrix and precision/recall/F1 for the fail class - what EDS
# actually cares about is catching bad wafers. The shared implementation
# lives in metrics.py; `evaluate` is imported above as print_evaluation.


if __name__ == "__main__":

    # whole runs are held out for testing, so the score measures
    # generalization to unseen process conditions (no leakage)
    X_train, y_train, X_test, y_test, test_runs = load_train_test_by_run()

    X_train, mean, std = standardize(X_train)
    X_test, _, _ = standardize(X_test, mean, std)

    print(f"Train {len(X_train)} wafers / Test {len(X_test)} wafers")
    print(f"Held-out test runs : {', '.join(test_runs)}")
    print(f"Fail ratio (train) : {(y_train == 0).mean() * 100:.2f}%")
    print()

    model = Perceptron(n_features=X_train.shape[1])
    model.fit(X_train, y_train, epochs=10)

    print()
    print("=== Test set evaluation ===")
    evaluate(y_test, model.predict(X_test))

    print()
    print("Learned linear boundary:")
    for name, w in zip(FEATURE_COLUMNS, model.w):
        print(f"  w[{name}] = {w:+.4f}")
    print(f"  b = {model.b:+.4f}")
