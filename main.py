import pandas as pd
import numpy as np
from simulation.wafer_generate import make_random_condi, add_measurement_noise
from simulation.wafer_analysis import wafer_analysis
from simulation.defect_analysis import extract_fail, analyze_Leakage_fail, analyze_Oxide_fail, analyze_Vth_fail
from simulation.visualization import show_vth_distribution, show_defect_pareto
from simulation.run_logger import save_run_info, save_wafer_data
from simulation.run_manage import setup_run
from simulation.config import PROCESS_CONFIG, PARAM_RANGES, MEASUREMENT_NOISE
from simulation.config_sampler import sample_config
from simulation.correlation_analysis import show_oxide_leakage_scatter, show_oxide_vth_scatter, calculate_correlation


def run_simulation(config, seed=None):

    run_path, time = setup_run()

    num_wafer = config["num_wafer"]

    Vth, Oxide, Leakage = make_random_condi(config)

    wafer_data = pd.DataFrame({
        "Wafer_ID" : range(1, num_wafer + 1),
        "Vth[V]" : Vth,
        "Oxide[nm]" : Oxide,
        "Leakage[nA]" : Leakage
    })

    print(wafer_data)

    check_yield = wafer_analysis(wafer_data)
    fail_wafer = extract_fail(wafer_data)
    fail_Vth = analyze_Vth_fail(wafer_data)
    fail_Oxide = analyze_Oxide_fail(wafer_data)
    fail_Leakage = analyze_Leakage_fail(wafer_data)

    # Pass/fail was judged on the true values above; from here on the
    # dataset holds what the tester measures (true + sensor noise),
    # because that is all a real fab - or an ML model - ever sees.
    Vth_m, Oxide_m, Leakage_m = add_measurement_noise(
        Vth, Oxide, Leakage, MEASUREMENT_NOISE
    )
    wafer_data["Vth[V]"] = Vth_m
    wafer_data["Oxide[nm]"] = Oxide_m
    wafer_data["Leakage[nA]"] = Leakage_m

    corr = calculate_correlation(wafer_data)

    print(f"Yield = {check_yield:.2f}%")
    print("Fail count :", len(fail_wafer))
    print("Vth error : ", fail_Vth)
    print("oxide error : ", fail_Oxide)
    print("leakage current error :", fail_Leakage)

    show_vth_distribution(wafer_data, run_path, time)

    show_defect_pareto(
        fail_Vth,
        fail_Oxide,
        fail_Leakage,
        run_path,
        time
    )

    save_run_info(
        run_path,
        config,
        check_yield,
        corr,
        seed
    )

    save_wafer_data(run_path, wafer_data)

    show_oxide_leakage_scatter(wafer_data, run_path, time)
    show_oxide_vth_scatter(wafer_data, run_path, time)

    return wafer_data, check_yield, corr


if __name__ == "__main__":
    config, seed = sample_config(PROCESS_CONFIG, PARAM_RANGES)
    run_simulation(config, seed)