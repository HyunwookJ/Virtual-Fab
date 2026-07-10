import json
from datetime import datetime


def save_run_info(
    run_path,
    process_config,
    yield_data,
    corr,
    seed=None
):

    info = {

        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),

        "Seed": seed,

        "Process_Config": process_config,

        "Yield": yield_data,

        "Correlation": {

            "Oxide-Vth":
                corr.loc["Oxide[nm]", "Vth[V]"],

            "Oxide-Leakage":
                corr.loc["Oxide[nm]", "Leakage[nA]"],

            "Vth-Leakage":
                corr.loc["Vth[V]", "Leakage[nA]"]
        }
    }

    with open(f"{run_path}/run_info.json", "w") as file:
        json.dump(info, file, indent=4)


# Save the per-wafer raw table (features + Result label) so that
# runs accumulate into an ML training dataset. Compressed to save disk.
def save_wafer_data(run_path, wafer_data):
    wafer_data.to_csv(f"{run_path}/wafers.csv.gz", index=False)