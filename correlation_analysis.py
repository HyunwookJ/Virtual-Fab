import matplotlib.pyplot as plt
from run_manage import time, run_path


def show_oxide_leakage_scatter(wafer_data):

    plt.scatter(
        wafer_data["Oxide[nm]"],
        wafer_data["Leakage[nA]"],
        s=1
    )

    plt.title("Oxide vs Leakage")
    plt.xlabel("Oxide Thickness [nm]")
    plt.ylabel("Leakage Current [nA]")

    plt.savefig(f"{run_path}/corr/{time}_corr_Oxide_Leakage.png")
    plt.close()

def show_oxide_vth_scatter(wafer_data):

    plt.scatter(
        wafer_data["Oxide[nm]"],
        wafer_data["Vth[V]"],
        s=1
    )

    plt.title("Oxide vs Vth")
    plt.xlabel("Oxide Thickness [nm]")
    plt.ylabel("Threshold Voltage [V]")

    plt.savefig(f"{run_path}/corr/{time}_corr_Oxide_Vth.png")
    plt.close()

def calculate_correlation(wafer_data):

    corr = wafer_data[
        [
            "Vth[V]",
            "Oxide[nm]",
            "Leakage[nA]"
        ]
    ].corr()

    return corr