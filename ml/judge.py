import os
import sys

import numpy as np

from dataset import load_run
from perceptron import standardize, evaluate
from mlp import MLP, MODEL_PATH


# The "EDS" step: take a trained model and grade a fresh wafer lot.
#
#   simulator (main.py) = the fab producing a new run
#   this script         = the inline tester binning each wafer
#
# The model only ever sees the measured features; the true Result
# label is used purely to score how well the model's verdict matches
# the ground-truth spec rule it was never given explicitly.
def judge_run(run_id, cutoff=0.5):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model at '{MODEL_PATH}'. "
            "Train one first: python3 ml/mlp.py"
        )

    model, mean, std, use_margins = MLP.load(MODEL_PATH)

    X, y = load_run(run_id)

    # the model must see wafers through the exact pipeline it was
    # trained on: same derived features, same scaling
    if use_margins:
        from features import add_margin_features
        X = add_margin_features(X)

    X_std, _, _ = standardize(X, mean, std)

    proba = model.predict_proba(X_std)          # P(good)
    pred = (proba >= cutoff).astype(int)

    print(f"=== Judging {run_id} ===")
    print(f"model       : {MODEL_PATH} (fail_weight={model.fail_weight:g}"
          f"{', margin features' if use_margins else ''})")
    print(f"wafers      : {len(y)}")
    print(f"cutoff      : P(good) >= {cutoff}")
    print()

    print("Model verdict vs ground-truth spec rule:")
    evaluate(y, pred)

    actual_yield = y.mean() * 100
    predicted_yield = pred.mean() * 100

    print()
    print(f"Yield  actual (rule)     : {actual_yield:.2f}%")
    print(f"       predicted (model) : {predicted_yield:.2f}%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python3 ml/judge.py <run_id> [cutoff]")
        print("example: python3 ml/judge.py run_011 0.5")
        sys.exit(1)

    run_id = sys.argv[1]
    cutoff = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    judge_run(run_id, cutoff)
