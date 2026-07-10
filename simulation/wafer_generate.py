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


# What the tester reports: true value + sensor noise.
# Leakage noise is multiplicative (lognormal) so it stays positive.
def add_measurement_noise(Vth, Oxide, Leakage, noise):

    n = len(Vth)

    Vth_measured = Vth + np.random.normal(0, noise["Vth"], n)
    Oxide_measured = Oxide + np.random.normal(0, noise["Oxide"], n)
    Leakage_measured = Leakage * np.random.lognormal(0, noise["Leakage"], n)

    return Vth_measured, Oxide_measured, Leakage_measured


