import argparse
import glob
import os
import subprocess
import sys

# Run the whole project in one command:
#   1. generate N simulation runs (training data)
#   2. train the MLP and save the model
#   3. generate one fresh lot and judge it with the saved model
#
# Usage:
#   python3 run_all.py                 # 10 runs, judge cutoff 0.5
#   python3 run_all.py --runs 20
#   python3 run_all.py --cutoff 0.9    # recall-first operating point

PY = sys.executable  # same interpreter (e.g. the venv) for all steps


def run(cmd, quiet=False):
    stdout = subprocess.DEVNULL if quiet else None
    subprocess.run(cmd, stdout=stdout, check=True)


def latest_run():
    runs = sorted(glob.glob("graph/run_*"))
    return os.path.basename(runs[-1]) if runs else None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end wafer pipeline: generate -> train -> judge"
    )
    parser.add_argument("--runs", type=int, default=10,
                        help="number of simulation runs to generate (default 10)")
    parser.add_argument("--cutoff", type=float, default=0.5,
                        help="P(good) threshold used when judging (default 0.5)")
    args = parser.parse_args()

    # must run from the repo root so graph/ and the ml/ scripts resolve
    if not os.path.exists("main.py"):
        sys.exit("run this from the repo root (where main.py lives)")

    print(f"[1/3] Generating {args.runs} simulation runs ...", flush=True)
    for i in range(1, args.runs + 1):
        print(f"      run {i}/{args.runs}", end="\r", flush=True)
        run([PY, "main.py"], quiet=True)
    print(flush=True)

    print("[2/3] Training MLP and saving the model ...", flush=True)
    run([PY, "ml/mlp.py"])

    print("\n[3/3] Generating one fresh lot and judging it ...", flush=True)
    run([PY, "main.py"], quiet=True)
    fresh = latest_run()
    print(f"      fresh lot: {fresh}\n", flush=True)
    run([PY, "ml/judge.py", fresh, str(args.cutoff)])


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"\npipeline step failed: {' '.join(e.cmd)}")
