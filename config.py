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