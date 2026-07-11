import numpy as np

from simulation.config import PHYSICS

# for conditions of wafer.
# physics selects which coupling constants the "world" obeys - the
# digital twin uses the default PHYSICS, the real fab (sim2real
# experiment) passes REAL_FAB_PHYSICS.
def make_random_condi(config, physics=PHYSICS):

    num_wafer = config["num_wafer"]

    Oxide = np.random.normal(
        loc=config["Oxide"]["mean"],
        scale=config["Oxide"]["std"],
        size=num_wafer
    )

    # CD (gate critical dimension) and process temperature are drawn
    # independently; the couplings below inject their physical effect
    # into Vth / Leakage, the same way Oxide already does.
    CD = np.random.normal(
        loc=config["CD"]["mean"],
        scale=config["CD"]["std"],
        size=num_wafer
    )

    Temp = np.random.normal(
        loc=config["Temp"]["mean"],
        scale=config["Temp"]["std"],
        size=num_wafer
    )

    Base_Vth = np.random.normal(
        loc=config["Vth"]["mean"],
        scale=config["Vth"]["std"],
        size=num_wafer
    )

    # thicker oxide raises Vth; smaller CD (shorter channel) lowers it
    Vth = (
        Base_Vth
        + physics["vth_oxide"] * (Oxide - 100)
        + physics["vth_cd"] * (CD - 45)
    )


    Base_Leakage = np.random.lognormal(
        mean=config["Leakage"]["mean"],
        sigma=config["Leakage"]["sigma"],
        size=num_wafer
    )

    # thinner oxide, smaller CD, and higher temperature all leak more
    Leakage = (
        Base_Leakage
        * np.exp(physics["leak_oxide"] * (Oxide - 100))
        * np.exp(physics["leak_cd"] * (CD - 45))
        * np.exp(physics["leak_temp"] * (Temp - 25))
    )

    return Vth, Oxide, Leakage, CD, Temp


# What the tester reports: true value + sensor noise.
# Leakage noise is multiplicative (lognormal) so it stays positive.
def add_measurement_noise(Vth, Oxide, Leakage, CD, Temp, noise):

    n = len(Vth)

    Vth_measured = Vth + np.random.normal(0, noise["Vth"], n)
    Oxide_measured = Oxide + np.random.normal(0, noise["Oxide"], n)
    Leakage_measured = Leakage * np.random.lognormal(0, noise["Leakage"], n)
    CD_measured = CD + np.random.normal(0, noise["CD"], n)
    Temp_measured = Temp + np.random.normal(0, noise["Temp"], n)

    return Vth_measured, Oxide_measured, Leakage_measured, CD_measured, Temp_measured


