import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# repo root is not on sys.path when this file is run as
# "python ml/visualize_boundary.py", but simulation.config lives there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import load_train_test_by_run
from perceptron import Perceptron, standardize
from mlp import MLP
from simulation.config import PROCESS_CONFIG

# Okabe-Ito blue/orange: a colorblind-safe pair
GOOD_COLOR = "#0072B2"
FAIL_COLOR = "#E69F00"


# Paint the model's verdict over a Vth-Oxide plane (Leakage held fixed
# at a healthy value), then overlay the true spec box. If the model
# really learned the rule, the painted "good" region should match the
# box; a single perceptron can only paint a half-plane.
def plot_decision_region(ax, model, mean, std, leakage_fixed, title):

    vth = np.linspace(0.63, 0.77, 400)
    oxide = np.linspace(95, 105, 400)
    VV, OO = np.meshgrid(vth, oxide)

    grid = np.column_stack([
        VV.ravel(),
        OO.ravel(),
        np.full(VV.size, leakage_fixed)
    ])
    grid_std = (grid - mean) / std

    region = model.predict(grid_std).reshape(VV.shape)

    ax.contourf(
        VV, OO, region,
        levels=[-0.5, 0.5, 1.5],
        colors=[FAIL_COLOR, GOOD_COLOR],
        alpha=0.20
    )

    vth_spec = PROCESS_CONFIG["Vth"]
    oxide_spec = PROCESS_CONFIG["Oxide"]

    ax.add_patch(plt.Rectangle(
        (vth_spec["spec_min"], oxide_spec["spec_min"]),
        vth_spec["spec_max"] - vth_spec["spec_min"],
        oxide_spec["spec_max"] - oxide_spec["spec_min"],
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=1.5,
        label="true spec box"
    ))

    ax.set_title(title)
    ax.set_xlabel("Vth [V] (measured)")


# A thin sample of real test wafers on top, colored by TRUE label,
# so we can see which wafers each region would misjudge.
def scatter_test_wafers(ax, X, y, n=3000, seed=0):

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    good = idx[y[idx] == 1]
    fail = idx[y[idx] == 0]

    ax.scatter(X[good, 0], X[good, 1], s=2, c=GOOD_COLOR,
               alpha=0.30, linewidths=0, label="good (true)")
    ax.scatter(X[fail, 0], X[fail, 1], s=3, c=FAIL_COLOR,
               alpha=0.55, linewidths=0, label="fail (true)")


if __name__ == "__main__":

    X_train, y_train, X_test, y_test, test_runs = load_train_test_by_run()

    X_train_std, mean, std = standardize(X_train)
    X_test_std, _, _ = standardize(X_test, mean, std)

    print("training single perceptron ...")
    perceptron = Perceptron(n_features=3)
    perceptron.fit(X_train_std, y_train, epochs=10)

    print()
    print("training MLP ...")
    mlp = MLP(n_features=3)
    mlp.fit(X_train_std, y_train, epochs=30)

    # slice the 3D feature space at a healthy leakage level,
    # so the Vth-Oxide plane shows the spec box cleanly
    leakage_fixed = float(np.median(X_train[:, 2]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    for ax, model, name in (
        (axes[0], perceptron, "Single perceptron"),
        (axes[1], mlp, "MLP (1 hidden layer)")
    ):
        acc = (model.predict(X_test_std) == y_test).mean()
        plot_decision_region(
            ax, model, mean, std, leakage_fixed,
            f"{name} - test accuracy {acc * 100:.2f}%"
        )
        scatter_test_wafers(ax, X_test, y_test)

    axes[0].set_ylabel("Oxide [nm] (measured)")
    axes[0].legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "Decision regions (shading = model verdict) "
        f"at Leakage = {leakage_fixed:.2f} nA"
    )

    os.makedirs("graph/ml", exist_ok=True)
    out_path = "graph/ml/decision_boundary.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    print()
    print(f"saved {out_path}")
