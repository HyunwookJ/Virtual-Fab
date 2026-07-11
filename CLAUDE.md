# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A semiconductor wafer-process simulator (a virtual fab) plus a from-scratch ML
classifier that learns to judge pass/fail from the simulated data, mimicking an
EDS test step. The perceptron, MLP, and backprop are all implemented directly in
numpy — **do not pull in scikit-learn, PyTorch, or any ML framework** for the
core models; that constraint is the point of the project.

## Working conventions

- **Do not run git commits.** Write and modify code only; leave `git commit`
  (and staging decisions) to the user, who prefers to handle commits themselves.

## Commands

Run everything from the repo root (scripts resolve `graph/` and `ml/` relative to
cwd, and several assert `main.py` exists).

```bash
source .venv/bin/activate            # venv already exists; deps in requirements.txt

python3 run_all.py                   # full pipeline: generate 10 runs -> train MLP -> judge a fresh lot
python3 run_all.py --runs 20         # change how many runs are generated
python3 run_all.py --cutoff 0.9      # recall-first operating point when judging

python3 main.py                      # generate ONE run (new run_XXX folder), append to dataset
python3 main.py --real               # generate a "real fab" run (shifted physics + noisier testers)
                                     # -> graph/real/run_XXX, used only by ml/sim2real.py
python3 ml/sim2real.py               # synthetic-pretrain vs real-budget experiment -> graph/ml/sim2real.png
python3 ml/dataset.py                # print current dataset status (runs / wafers / good / fail)
python3 ml/perceptron.py             # train + eval single-layer perceptron (pocket)
python3 ml/mlp.py                    # train + eval MLP, saves model to graph/ml/mlp_model.npz
python3 ml/mlp.py --hidden 32 16 --l2 1e-4 --epochs 50   # deeper net / hyperparams via CLI
python3 ml/mlp.py --margins          # train on derived margin features (flag saved into the model;
                                     # judge.py reads it and applies the same transform automatically)
python3 ml/perceptron.py --margins   # margin features rescue the perceptron (see ml/features.py)
python3 ml/judge.py run_011          # judge one run with the saved model (cutoff 0.5)
python3 ml/judge.py run_011 0.9      # judge with a custom cutoff
python3 ml/visualize_boundary.py     # decision-boundary figure -> graph/ml/decision_boundary.png
python3 ml/visualize_metrics.py      # ROC/PR/training-curve/confusion -> graph/ml/model_evaluation.png
```

There is no test suite, linter, or build step. Verification is done by running the
scripts and reading their printed metrics.

## Architecture

Two decoupled stages connected only through files on disk under `graph/` (which
is gitignored — all of it is regenerable):

1. **`simulation/`** — the virtual fab. `main.py` calls `run_simulation()`, which
   creates a new `graph/run_XXX/` folder and writes `wafers.csv.gz` (per-wafer
   rows) + `run_info.json` (the process condition and seed). Each `main.py`
   invocation is one run and *appends* a new folder; data accumulates across runs.
2. **`ml/`** — reads every `graph/run_*/wafers.csv.gz` back via `ml/dataset.py`,
   trains, and (for `judge.py`) writes/reads the saved model at
   `graph/ml/mlp_model.npz`.

### Two subtleties that drive the whole design

- **Labels are true-value based; features are measured.** In `main.py`,
  `wafer_analysis()` computes `Result` (pass/fail) from the *true* physical values
  first; only afterward is sensor noise added and the feature columns
  (`Vth[V]`, `Oxide[nm]`, `Leakage[nA]`, `CD[nm]`, `Temp[C]`) overwritten with the
  noisy measurements.
  So the CSV holds noisy features but clean labels. This is deliberate: near a
  spec boundary the measurement can't determine pass/fail, making the problem
  genuinely probabilistic — no model can reach 100%. If you touch the ordering of
  labeling vs. noise in `main.py`, you break this property. Noise magnitudes live
  in `MEASUREMENT_NOISE` in `simulation/config.py`.

- **Split by run, not by wafer.** `load_train_test_by_run()` holds out whole runs
  for the test set. All wafers in a run share one process condition (sampled per
  run by `config_sampler.py`), so splitting by wafer would leak conditions into
  the test set. Keep evaluation run-based when adding models.

### Spec / config

`simulation/config.py` is the single source of truth. `PROCESS_CONFIG` holds spec
limits (fixed — they're product requirements) and per-run *default* center/spread.
`PARAM_RANGES` gives the `(low, high)` bounds from which `config_sampler.py`
randomizes each run's process center/spread, producing run-to-run variation. The
"good" region is a spec **box** (an AND of per-parameter ranges) — that's why a
single-layer perceptron structurally can't solve it and the MLP can.

The feature/label column names are contracts between the two stages, defined once
as `FEATURE_COLUMNS` / `LABEL_COLUMN` in `ml/dataset.py` and matching the columns
written by the simulation. Changing a column name means changing both sides.

The sim2real experiment (`ml/sim2real.py`) treats the default simulator as a
digital twin and a second world — `REAL_FAB_PHYSICS` / `REAL_FAB_NOISE` in
`simulation/config.py`, generated via `main.py --real` into `graph/real/run_*` —
as the real fab. Physical coupling constants live in `PHYSICS` (config.py) and are
injected into `make_random_condi`; spec limits are shared between both worlds.

Derived margin features (`ml/features.py`, opt-in via `--margins`) are computed at
train/judge time from the measured columns plus spec limits — never stored in the
CSV. Whether a saved model expects them is persisted as `use_margins` inside
`mlp_model.npz`; `judge.py` reads that flag and applies the identical transform,
so the CSV contract stays five raw columns regardless.
