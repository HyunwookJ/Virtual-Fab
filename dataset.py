import glob
import os

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


if __name__ == "__main__":
    data = load_all_runs()

    print(data)
    print()
    print("Runs :", data["Run"].nunique())
    print("Wafers :", len(data))
    print("Good :", int(data[LABEL_COLUMN].sum()),
          "| Fail :", int((~data[LABEL_COLUMN].astype(bool)).sum()))
