import numpy as np

from simulation.config import PHYSICS

# for conditions of wafer.
# physics selects which coupling constants the "world" obeys - the
# digital twin uses the default PHYSICS, the real fab (sim2real
# experiment) passes REAL_FAB_PHYSICS.
# rng: the run's random stream (see config_sampler.sample_config). Passing
# one in makes the run reproducible from its recorded seed; omitting it
# falls back to a fresh unseeded stream.
def make_random_condi(config, physics=PHYSICS, rng=None):

    rng = np.random.default_rng() if rng is None else rng

    num_wafer = config["num_wafer"]

    Oxide = rng.normal(
        loc=config["Oxide"]["mean"],
        scale=config["Oxide"]["std"],
        size=num_wafer
    )

    # CD (gate critical dimension) and process temperature are drawn
    # independently; the couplings below inject their physical effect
    # into Vth / Leakage, the same way Oxide already does.
    CD = rng.normal(
        loc=config["CD"]["mean"],
        scale=config["CD"]["std"],
        size=num_wafer
    )

    Temp = rng.normal(
        loc=config["Temp"]["mean"],
        scale=config["Temp"]["std"],
        size=num_wafer
    )

    Base_Vth = rng.normal(
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


    Base_Leakage = rng.lognormal(
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
# rng: same run stream as make_random_condi, so the measurements are part
# of what the run's seed reproduces.
def add_measurement_noise(Vth, Oxide, Leakage, CD, Temp, noise, rng=None):

    rng = np.random.default_rng() if rng is None else rng

    n = len(Vth)

    Vth_measured = Vth + rng.normal(0, noise["Vth"], n)
    Oxide_measured = Oxide + rng.normal(0, noise["Oxide"], n)
    Leakage_measured = Leakage * rng.lognormal(0, noise["Leakage"], n)
    CD_measured = CD + rng.normal(0, noise["CD"], n)
    Temp_measured = Temp + rng.normal(0, noise["Temp"], n)

    return Vth_measured, Oxide_measured, Leakage_measured, CD_measured, Temp_measured


