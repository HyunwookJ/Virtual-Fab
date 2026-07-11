from simulation.config import PROCESS_CONFIG

def wafer_analysis(wafer_data):

    condition = (
        (wafer_data["Vth[V]"] >= PROCESS_CONFIG["Vth"]["spec_min"]) &
        (wafer_data["Vth[V]"] <= PROCESS_CONFIG["Vth"]["spec_max"]) &
        (wafer_data["Oxide[nm]"] >= PROCESS_CONFIG["Oxide"]["spec_min"]) &
        (wafer_data["Oxide[nm]"] <= PROCESS_CONFIG["Oxide"]["spec_max"]) &
        (wafer_data["Leakage[nA]"] <= PROCESS_CONFIG["Leakage"]["spec_max"]) &
        (wafer_data["CD[nm]"] >= PROCESS_CONFIG["CD"]["spec_min"]) &
        (wafer_data["CD[nm]"] <= PROCESS_CONFIG["CD"]["spec_max"]) &
        (wafer_data["Temp[C]"] >= PROCESS_CONFIG["Temp"]["spec_min"]) &
        (wafer_data["Temp[C]"] <= PROCESS_CONFIG["Temp"]["spec_max"])
    )

    wafer_data["Result"] = condition

    yield_data = (
        wafer_data["Result"].sum() / len(wafer_data) * 100
    )

    return yield_data