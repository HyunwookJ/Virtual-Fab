from simulation.config import PROCESS_CONFIG

def extract_fail(wafer_data):
    fail_wafer = wafer_data[wafer_data["Result"] == False]

    return fail_wafer

def analyze_Vth_fail(wafer_data):
    fail = (
        (wafer_data["Vth[V]"] < PROCESS_CONFIG["Vth"]["spec_min"]) |
        (wafer_data["Vth[V]"] > PROCESS_CONFIG["Vth"]["spec_max"])
    )

    return fail.sum()

def analyze_Oxide_fail(wafer_data):
    fail = (
        (wafer_data["Oxide[nm]"] < PROCESS_CONFIG["Oxide"]["spec_min"]) |
        (wafer_data["Oxide[nm]"] > PROCESS_CONFIG["Oxide"]["spec_max"])
    )

    return fail.sum()

def analyze_Leakage_fail(wafer_data):
    fail = (
        (wafer_data["Leakage[nA]"] > PROCESS_CONFIG["Leakage"]["spec_max"])
    )

    return fail.sum()