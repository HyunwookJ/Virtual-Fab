import pandas as pd
import numpy as np
from simulation.wafer_generate import make_random_condi, add_measurement_noise
from simulation.wafer_analysis import wafer_analysis
from simulation.defect_analysis import extract_fail, analyze_Leakage_fail, analyze_Oxide_fail, analyze_Vth_fail, analyze_CD_fail, analyze_Temp_fail
from simulation.visualization import show_vth_distribution, show_defect_pareto
from simulation.run_logger import save_run_info, save_wafer_data
from simulation.run_manage import setup_run
from simulation.config import (
    PROCESS_CONFIG, PARAM_RANGES, MEASUREMENT_NOISE,
    PHYSICS, REAL_FAB_PHYSICS, REAL_FAB_NOISE,
)
from simulation.config_sampler import sample_config
from simulation.correlation_analysis import show_oxide_leakage_scatter, show_oxide_vth_scatter, calculate_correlation


def run_simulation(config, seed=None, physics=PHYSICS,
                   noise=MEASUREMENT_NOISE, base_dir="graph", rng=None):

    run_path, time = setup_run(base_dir)

    num_wafer = config["num_wafer"]

    # rng carries the run's random stream (from sample_config), so the seed
    # recorded in run_info.json reproduces the wafers, not just the process
    # condition. Without it the run is still valid - just not repeatable.
    Vth, Oxide, Leakage, CD, Temp = make_random_condi(config, physics, rng)

    wafer_data = pd.DataFrame({
        "Wafer_ID" : range(1, num_wafer + 1),
        "Vth[V]" : Vth,
        "Oxide[nm]" : Oxide,
        "Leakage[nA]" : Leakage,
        "CD[nm]" : CD,
        "Temp[C]" : Temp
    })

    check_yield = wafer_analysis(wafer_data)
    fail_wafer = extract_fail(wafer_data)
    fail_Vth = analyze_Vth_fail(wafer_data)
    fail_Oxide = analyze_Oxide_fail(wafer_data)
    fail_Leakage = analyze_Leakage_fail(wafer_data)
    fail_CD = analyze_CD_fail(wafer_data)
    fail_Temp = analyze_Temp_fail(wafer_data)

    # Pass/fail was judged on the true values above; from here on the
    # dataset holds what the tester measures (true + sensor noise),
    # because that is all a real fab - or an ML model - ever sees.
    Vth_m, Oxide_m, Leakage_m, CD_m, Temp_m = add_measurement_noise(
        Vth, Oxide, Leakage, CD, Temp, noise, rng
    )
    wafer_data["Vth[V]"] = Vth_m
    wafer_data["Oxide[nm]"] = Oxide_m
    wafer_data["Leakage[nA]"] = Leakage_m
    wafer_data["CD[nm]"] = CD_m
    wafer_data["Temp[C]"] = Temp_m

    corr = calculate_correlation(wafer_data)

    print(f"Yield = {check_yield:.2f}%")
    print("Fail count :", len(fail_wafer))
    print("Vth error : ", fail_Vth)
    print("oxide error : ", fail_Oxide)
    print("leakage current error :", fail_Leakage)
    print("CD error : ", fail_CD)
    print("temperature error : ", fail_Temp)

    show_vth_distribution(wafer_data, run_path, time)

    show_defect_pareto(
        fail_Vth,
        fail_Oxide,
        fail_Leakage,
        fail_CD,
        fail_Temp,
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
    import argparse

    parser = argparse.ArgumentParser(description="Generate one wafer run")
    parser.add_argument(
        "--real", action="store_true",
        help="generate with the REAL fab's physics/noise (sim2real "
             "experiment) -> graph/real/run_XXX instead of graph/run_XXX"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed for this run. Reproduces the process condition AND the "
             "wafers generated under it. Omit for a fresh random run (the "
             "drawn seed is still recorded in run_info.json)."
    )
    args = parser.parse_args()

    config, seed, rng = sample_config(PROCESS_CONFIG, PARAM_RANGES, args.seed)

    print(f"seed : {seed}")

    if args.real:
        run_simulation(config, seed, rng=rng,
                       physics=REAL_FAB_PHYSICS,
                       noise=REAL_FAB_NOISE,
                       base_dir="graph/real")
    else:
        run_simulation(config, seed, rng=rng)