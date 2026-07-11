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