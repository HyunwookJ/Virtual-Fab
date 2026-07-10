import glob
import os

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["Vth[V]", "Oxide[nm]", "Leakage[nA]"]
LABEL_COLUMN = "Result"


# Collect every run's wafers.csv.gz under graph_dir into one table.
# A "Run" column is added so each wafer can be traced back to the
# process condition (run_info.json) it came from.
def load_all_runs(graph_dir="graph"):

    paths = sorted(glob.glob(f"{graph_dir}/run_*/wafers.csv.gz"))

    if not paths:
        raise FileNotFoundError(
            f"No wafers.csv.gz found under '{graph_dir}/'. "
            "Run main.py first to generate data."
        )

    frames = []

    for path in paths:
        df = pd.read_csv(path)
        df["Run"] = os.path.basename(os.path.dirname(path))
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# Return (X, y) ready for model training:
#   X = per-wafer features [Vth, Oxide, Leakage]
#   y = pass/fail label (1 = good, 0 = fail)
def load_dataset(graph_dir="graph"):

    data = load_all_runs(graph_dir)

    X = data[FEATURE_COLUMNS].to_numpy()
    y = data[LABEL_COLUMN].astype(int).to_numpy()

    return X, y


# Load one run's wafers as (X, y). Used when grading a single fresh
# lot with a trained model (see ml/judge.py).
def load_run(run_id, graph_dir="graph"):

    path = f"{graph_dir}/{run_id}/wafers.csv.gz"

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run 'python3 main.py' to create runs."
        )

    df = pd.read_csv(path)

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[LABEL_COLUMN].astype(int).to_numpy()

    return X, y


# Split by RUN, not by wafer. Wafers from the same run share one
# process condition; if they land in both train and test, the test
# score partly measures memorization of already-seen conditions
# (data leakage). Holding out whole runs asks the question production
# actually cares about: does the model generalize to process
# conditions it has never seen?
def load_train_test_by_run(test_fraction=0.2, graph_dir="graph", seed=42):

    data = load_all_runs(graph_dir)

    runs = sorted(data["Run"].unique())

    rng = np.random.default_rng(seed)
    rng.shuffle(runs)

    n_test = max(1, round(len(runs) * test_fraction))
    test_runs = sorted(runs[:n_test])

    is_test = data["Run"].isin(test_runs)

    train = data[~is_test]
    test = data[is_test]

    X_train = train[FEATURE_COLUMNS].to_numpy()
    y_train = train[LABEL_COLUMN].astype(int).to_numpy()
    X_test = test[FEATURE_COLUMNS].to_numpy()
    y_test = test[LABEL_COLUMN].astype(int).to_numpy()

    return X_train, y_train, X_test, y_test, test_runs


if __name__ == "__main__":
    data = load_all_runs()

    print(data)
    print()
    print("Runs :", data["Run"].nunique())
    print("Wafers :", len(data))
    print("Good :", int(data[LABEL_COLUMN].sum()),
          "| Fail :", int((~data[LABEL_COLUMN].astype(bool)).sum()))
