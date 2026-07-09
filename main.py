import pandas as pd
import numpy as np
from wafer_generate import make_random_condi
from wafer_analysis import wafer_analysis
from defect_analysis import extract_fail, analyze_Leakage_fail, analyze_Oxide_fail, analyze_Vth_fail
from visualization import show_vth_distribution, show_defect_pareto
from run_logger import save_run_info
from run_manage import run_path
from config import PROCESS_CONFIG
from correlation_analysis import show_oxide_leakage_scatter, show_oxide_vth_scatter, calculate_correlation


num_wafer = PROCESS_CONFIG["num_wafer"]

Vth, Oxide, Leakage = make_random_condi(PROCESS_CONFIG)

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

corr = calculate_correlation(wafer_data)

print(f"Yield = {check_yield:.2f}%")
print("Fail count :", len(fail_wafer))
print("Vth error : ", fail_Vth)
print("oxide error : ", fail_Oxide)
print("leakage current error :", fail_Leakage)

show_vth_distribution(wafer_data)

show_defect_pareto(
    fail_Vth,
    fail_Oxide,
    fail_Leakage
)

save_run_info(
    run_path,
    PROCESS_CONFIG,
    check_yield,
    corr
)


show_oxide_leakage_scatter(wafer_data)
show_oxide_vth_scatter(wafer_data)