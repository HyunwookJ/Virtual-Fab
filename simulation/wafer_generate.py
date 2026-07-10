import numpy as np

# for conditions of wafer
def make_random_condi(config):

    num_wafer = config["num_wafer"]

    Oxide = np.random.normal(
        loc=config["Oxide"]["mean"],
        scale=config["Oxide"]["std"],
        size=num_wafer
    )

    Base_Vth = np.random.normal(
        loc=config["Vth"]["mean"],
        scale=config["Vth"]["std"],
        size=num_wafer
    )

    Vth = (
        Base_Vth + 0.01 * (Oxide - 100)
    )


    Base_Leakage = np.random.lognormal(
        mean=config["Leakage"]["mean"],
        sigma=config["Leakage"]["sigma"],
        size=num_wafer
    )

    Leakage = (
        Base_Leakage * np.exp(-0.3 * (Oxide - 100))
    )

    return Vth, Oxide, Leakage


