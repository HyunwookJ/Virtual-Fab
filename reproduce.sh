#!/usr/bin/env bash
# Regenerate the exact dataset behind the numbers in the README.
#
# Every run is generated from a fixed seed, and that seed fixes both the
# process condition and the 20,000 wafers drawn under it - so this script
# reproduces the dataset bit for bit on any machine.
#
#   ./reproduce.sh          # 12 twin runs + 3 real-fab runs
#
# graph/ is gitignored (the data is regenerable - that is the point of
# this script), so run this once after cloning, then run the experiments
# in README Appendix A.
set -euo pipefail

SYNTHETIC_SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12)
REAL_SEEDS=(101 102 103)

echo "=== digital twin: ${#SYNTHETIC_SEEDS[@]} runs -> graph/run_XXX ==="
for s in "${SYNTHETIC_SEEDS[@]}"; do
    echo "  seed $s"
    python3 main.py --seed "$s" > /dev/null
done

echo "=== real fab: ${#REAL_SEEDS[@]} runs -> graph/real/run_XXX ==="
for s in "${REAL_SEEDS[@]}"; do
    echo "  seed $s"
    python3 main.py --real --seed "$s" > /dev/null
done

echo "done. Next: python3 ml/mlp.py, python3 ml/sim2real.py (see README)."
