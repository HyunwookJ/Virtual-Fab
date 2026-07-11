# Before data processing, configuare the initial condition out.

PROCESS_CONFIG = {
    "num_wafer": 20000,

    "Vth": {
        "mean": 0.70,
        "std": 0.008,
        "spec_min": 0.67,
        "spec_max": 0.73
    },

    "Oxide": {
        "mean": 100,
        "std": 0.8,
        "spec_min": 97,
        "spec_max": 103
    },

    "Leakage": {
        "mean": 1.0,
        "sigma": 0.15,
        "spec_max": 10
    },

    "CD": {
        "mean": 45,
        "std": 0.6,
        "spec_min": 43.5,
        "spec_max": 46.5
    },

    "Temp": {
        "mean": 25,
        "std": 0.8,
        "spec_min": 23,
        "spec_max": 27
    }
}


# Physical couplings used by wafer_generate.py: how strongly one true
# quantity drags another. These ARE the simulator's physics.
PHYSICS = {
    "vth_oxide": 0.01,    # dVth per nm of oxide above 100
    "vth_cd": 0.004,      # dVth per nm of CD above 45
    "leak_oxide": -0.3,   # log-leakage slope vs oxide
    "leak_cd": -0.05,     # log-leakage slope vs CD
    "leak_temp": 0.03     # log-leakage slope vs temperature
}


# --- The "real fab" (sim2real experiment) ---------------------------
# Our simulator above plays the role of a DIGITAL TWIN: a physics model
# someone built from theory. The real fab obeys physics the twin gets
# slightly wrong (stronger couplings) and its testers are noisier.
# Neither is ever visible to the twin - that mismatch is the sim2real
# gap that ml/sim2real.py measures. Spec limits stay identical: they
# are product requirements, not physics.
REAL_FAB_PHYSICS = {
    "vth_oxide": 0.013,
    "vth_cd": 0.006,
    "leak_oxide": -0.36,
    "leak_cd": -0.03,
    "leak_temp": 0.05
}

REAL_FAB_NOISE = {
    "Vth": 0.007,
    "Oxide": 0.7,
    "Leakage": 0.14,
    "CD": 0.42,
    "Temp": 0.3
}


# Measurement noise of the tester (the probe is not perfect).
# Pass/fail (Result) is judged on the TRUE physical values, but the
# dataset stores what the tester MEASURES: true value + sensor noise.
# Near a spec boundary the measurement can no longer tell good from
# bad, so no classifier can reach 100% - this is what makes the
# problem realistic instead of "rediscovering a known rule".
MEASUREMENT_NOISE = {
    "Vth": 0.005,     # additive gaussian std [V]
    "Oxide": 0.5,     # additive gaussian std [nm]
    "Leakage": 0.10,  # multiplicative lognormal sigma
    "CD": 0.3,        # additive gaussian std [nm]
    "Temp": 0.4       # additive gaussian std [C]
}


# How each run's process condition is randomized.
# Only the process center/spread (mean, std, sigma) drift run-to-run;
# spec limits stay fixed because they are product requirements.
# Each entry is a (low, high) range sampled uniformly.
PARAM_RANGES = {

    "Vth": {
        "mean": (0.685, 0.715),
        "std":  (0.006, 0.015)
    },

    "Oxide": {
        "mean": (98.5, 101.5),
        "std":  (0.6, 1.5)
    },

    "Leakage": {
        "mean":  (0.8, 1.3),
        "sigma": (0.10, 0.25)
    },

    "CD": {
        "mean": (44.4, 45.6),
        "std":  (0.5, 0.85)
    },

    "Temp": {
        "mean": (24.3, 25.7),
        "std":  (0.7, 1.1)
    }
}