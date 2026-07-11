import matplotlib.pyplot as plt


def show_vth_distribution(wafer_data, run_path, time):
    plt.hist(
        wafer_data["Vth[V]"],
        bins=30
    )


    plt.title("Vth Distribution")
    plt.xlabel("Vth [V]")
    plt.ylabel("Number of Wafers")

    plt.savefig(f"{run_path}/distribution/{time}_Vth_distribution.png")
    plt.close()

def show_defect_pareto(Vth_fail, Oxide_fail, Leakage_fail, CD_fail, Temp_fail,
                       run_path, time):

    causes = [
        "Vth",
        "Oxide",
        "Leakage",
        "CD",
        "Temp"
    ]

    counts = [
        Vth_fail,
        Oxide_fail,
        Leakage_fail,
        CD_fail,
        Temp_fail
    ]

    plt.bar(causes, counts)

    plt.title("Defect Cause Pareto Chart")
    plt.xlabel("Failure Cause")
    plt.ylabel("Number of Wafers")

    plt.savefig(f"{run_path}/pareto/{time}_Defect_Pareto.png")
    plt.close()