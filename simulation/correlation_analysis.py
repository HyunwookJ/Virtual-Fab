import matplotlib.pyplot as plt


def show_oxide_leakage_scatter(wafer_data, run_path, time):

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

def show_oxide_vth_scatter(wafer_data, run_path, time):

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
            "Leakage[nA]",
            "CD[nm]",
            "Temp[C]"
        ]
    ].corr()

    return corr