import os

import numpy as np
import matplotlib.pyplot as plt

from dataset import load_train_test_by_run
from perceptron import Perceptron, standardize
from mlp import MLP
from metrics import roc_curve, pr_curve, binary_metrics

# Okabe-Ito colorblind-safe palette
MLP_COLOR = "#0072B2"     # blue
PERC_COLOR = "#E69F00"    # orange
REF_COLOR = "#999999"     # grey reference lines


def plot_roc(ax, curves):
    for name, (fpr, tpr, auc), color in curves:
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=REF_COLOR, lw=1, label="random")
    ax.set_xlabel("False positive rate (good wafers rejected)")
    ax.set_ylabel("True positive rate (fails caught)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right", fontsize=8)


def plot_pr(ax, curves, prevalence):
    for name, (rec, prec, ap), color in curves:
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP {ap:.3f})")
    ax.axhline(prevalence, ls="--", color=REF_COLOR, lw=1,
               label=f"baseline ({prevalence:.2f})")
    ax.set_xlabel("Recall (fails caught)")
    ax.set_ylabel("Precision (flagged that truly fail)")
    ax.set_title("Precision-Recall curve (fail class)")
    ax.legend(loc="lower left", fontsize=8)


# The whole reason to track a held-out curve: if train loss keeps
# dropping while test loss turns back up, the model is memorizing.
def plot_training_curve(ax, history):
    epochs = [h["epoch"] for h in history]
    ax.plot(epochs, [h["train_loss"] for h in history],
            color=MLP_COLOR, lw=2, label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history],
            color=PERC_COLOR, lw=2, label="test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title("MLP training vs test loss")
    ax.legend(loc="upper right", fontsize=8)


def plot_confusion(ax, y_true, y_pred):
    m = binary_metrics(y_true, y_pred)
    # rows = true (fail, good), cols = predicted (fail, good)
    matrix = np.array([[m["tp"], m["fn"]],
                       [m["fp"], m["tn"]]])
    ax.imshow(matrix, cmap="Blues")
    labels = [["TP", "FN"], ["FP", "TN"]]
    thresh = matrix.max() / 2
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{labels[r][c]}\n{matrix[r, c]:,}",
                    ha="center", va="center", fontsize=10,
                    color="white" if matrix[r, c] > thresh else "black")
    ax.set_xticks([0, 1], ["pred fail", "pred good"])
    ax.set_yticks([0, 1], ["true fail", "true good"])
    ax.set_title(f"MLP confusion @ cutoff 0.5 "
                 f"(recall {m['recall'] * 100:.1f}%)")


if __name__ == "__main__":

    X_train, y_train, X_test, y_test, test_runs = load_train_test_by_run()

    X_train_std, mean, std = standardize(X_train)
    X_test_std, _, _ = standardize(X_test, mean, std)

    print("training single perceptron ...")
    perceptron = Perceptron(n_features=X_train_std.shape[1])
    perceptron.fit(X_train_std, y_train, epochs=10)

    print()
    print("training MLP (weighted) ...")
    mlp = MLP(n_features=X_train_std.shape[1], fail_weight=5.0)
    history = mlp.fit(X_train_std, y_train, epochs=30,
                      X_val=X_test_std, y_val=y_test)

    # good-scores for ranking: P(good) for the MLP, signed distance for
    # the perceptron (it has no probability, only a side of the line)
    mlp_score = mlp.predict_proba(X_test_std)
    perc_score = perceptron.decision_function(X_test_std)

    roc_curves = [
        ("MLP", roc_curve(y_test, mlp_score), MLP_COLOR),
        ("Perceptron", roc_curve(y_test, perc_score), PERC_COLOR),
    ]
    pr_curves = [
        ("MLP", pr_curve(y_test, mlp_score), MLP_COLOR),
        ("Perceptron", pr_curve(y_test, perc_score), PERC_COLOR),
    ]
    prevalence = float((y_test == 0).mean())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_roc(axes[0, 0], roc_curves)
    plot_pr(axes[0, 1], pr_curves, prevalence)
    plot_training_curve(axes[1, 0], history)
    plot_confusion(axes[1, 1], y_test, mlp.predict(X_test_std))

    fig.suptitle(
        "Model evaluation on held-out runs "
        f"({', '.join(test_runs)}) - fail = positive class",
        fontsize=13
    )

    os.makedirs("graph/ml", exist_ok=True)
    out_path = "graph/ml/model_evaluation.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    print()
    print(f"ROC AUC  - MLP {roc_curves[0][1][2]:.3f} | "
          f"Perceptron {roc_curves[1][1][2]:.3f}")
    print(f"PR  AP   - MLP {pr_curves[0][1][2]:.3f} | "
          f"Perceptron {pr_curves[1][1][2]:.3f}")
    print(f"saved {out_path}")
