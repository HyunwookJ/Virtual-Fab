import os
import sys

import numpy as np

# repo root is not on sys.path when run as "python ml/<script>.py",
# but simulation.config lives there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FEATURE_COLUMNS
from simulation.config import PROCESS_CONFIG

# Derived features: signed distance from each measurement to its
# NEAREST spec edge, normalized so every margin lives on a comparable
# scale (~[-1, 1] inside the box). Positive = inside spec, negative =
# outside, 0 = exactly on the edge.
#
# This is feature engineering as prior knowledge: the spec limits are
# not secrets (a fab knows its own product requirements), so we may
# hand the model the *geometry* of the rule and let it learn only the
# judgment. The good region "all margins >= 0" is much easier to
# express than a raw 5-D box -- with the extra MinMargin feature it
# even becomes a single linear cut, which is exactly why the perceptron
# benefits the most. The problem still is not trivial: margins are
# computed from MEASURED values while labels come from TRUE values, so
# near-edge wafers remain genuinely ambiguous.

_COLUMN_GROUP = {
    "Vth[V]": "Vth",
    "Oxide[nm]": "Oxide",
    "Leakage[nA]": "Leakage",
    "CD[nm]": "CD",
    "Temp[C]": "Temp",
}

MARGIN_COLUMNS = [f"Margin({_COLUMN_GROUP[c]})" for c in FEATURE_COLUMNS]
MIN_MARGIN_COLUMN = "MinMargin"

# column names of the expanded matrix returned by add_margin_features
EXTENDED_COLUMNS = FEATURE_COLUMNS + MARGIN_COLUMNS + [MIN_MARGIN_COLUMN]


def add_margin_features(X):

    X = np.asarray(X, dtype=float)
    margins = []

    for j, col in enumerate(FEATURE_COLUMNS):

        spec = PROCESS_CONFIG[_COLUMN_GROUP[col]]
        x = X[:, j]

        if "spec_min" in spec:
            # two-sided spec: distance to the nearer edge, in units of
            # the spec half-width (1.0 = dead center, <0 = out of spec)
            half = (spec["spec_max"] - spec["spec_min"]) / 2
            m = np.minimum(x - spec["spec_min"], spec["spec_max"] - x) / half
        else:
            # one-sided spec (Leakage): distance below the ceiling,
            # in units of the ceiling itself
            m = (spec["spec_max"] - x) / spec["spec_max"]

        margins.append(m)

    margins = np.column_stack(margins)

    # the wafer is (measured-)good iff its WORST margin is >= 0, so the
    # min over margins linearizes the whole spec-box rule into one axis
    min_margin = margins.min(axis=1, keepdims=True)

    return np.hstack([X, margins, min_margin])
