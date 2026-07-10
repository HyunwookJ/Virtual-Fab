import copy
import numpy as np


# Draw a fresh process condition for one run.
#
# base_config : the nominal PROCESS_CONFIG (spec limits, num_wafer, ...)
# param_ranges: PARAM_RANGES describing which keys to randomize and their bounds
# seed        : optional int for reproducibility. If None, an OS-drawn seed is
#               used and returned so the exact run can be reproduced later.
#
# Returns (config, used_seed).
def sample_config(base_config, param_ranges, seed=None):

    seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seq)

    config = copy.deepcopy(base_config)

    for group, params in param_ranges.items():
        for key, (low, high) in params.items():
            config[group][key] = float(rng.uniform(low, high))

    used_seed = seq.entropy

    return config, used_seed
