# Before data processing, configuare the initial condition out.

PROCESS_CONFIG = {
    "num_wafer": 100000,

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
    }
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
    }
}