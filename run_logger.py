import json
from datetime import datetime


def save_run_info(
    run_path,
    process_config,
    yield_data,
    corr
):

    info = {
        
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),

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