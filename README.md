# Virtual Fab — Learning a Wafer Pass/Fail Classifier from Simulated Process Data

*Can cheap synthetic wafer data replace expensive real wafers for training an EDS pass/fail classifier?*

A record of an experiment in which a semiconductor wafer-process simulator is built, and the data it generates are used to train a good/fail classifier for the EDS test step. The simulator and the classifiers (perceptron, MLP, backpropagation) are implemented entirely in numpy, without any external ML framework.

> 한국어 원문은 [`README.ko.md`](README.ko.md)에 있다.

## 1. Introduction

Obtaining labeled training data in a real fab runs into a cost problem. The unit cost of a single wafer is high, producing one row of data requires physically passing a wafer through dozens of process and measurement steps, and the fail samples that training needs become rarer the higher the yield. Collecting data across varying process conditions is even more expensive than simply increasing the wafer count.

As an alternative, this project examines whether it is viable to train a classifier on data mass-produced at low cost by a simulator (a digital twin) that approximates the process physics. Three concrete questions are posed.

1. Is it possible to learn pass/fail judgment from simulated data, and what model capacity does the structure of the problem require? (Sections 2–4.3)
2. Do the data remain useful even when the simulator's physics is mismatched with reality? (Section 4.4)
3. If so, to what extent can they replace real data? (Section 4.4)

## 2. Simulator Design

### 2.1 Data generation

Each run of the simulator generates one run of 20,000 wafers. There are five process parameters — threshold voltage Vth[V], oxide thickness Oxide[nm], leakage current Leakage[nA], gate critical dimension CD[nm], and process temperature Temp[C] — and for each run the center and spread of every parameter are randomly sampled within fixed ranges (`PARAM_RANGES` in `simulation/config.py`), reproducing run-to-run process variation.

Physical correlations are imposed among the parameters. A thicker oxide raises Vth and reduces leakage; a smaller CD (short channel) lowers Vth and increases leakage; and a higher temperature increases leakage exponentially. The coupling coefficients are defined in `PHYSICS`.

### 2.2 Separating labels from measurements

The pass/fail label (Result) is decided by whether every parameter's *true* value satisfies its spec interval (an AND over a five-dimensional spec box). The feature values stored in the dataset, however, are not the true values but the measured values — the true values plus sensor noise — because a measured value is all a real fab's tester ever reports.

This design makes wafers near a spec boundary impossible to classify from measurement alone, turning the task into an inherently probabilistic classification problem in which no model can reach 100% accuracy. The ordering in `main.py` — label from the true values first, then add noise — guarantees this property.

### 2.3 Evaluation protocol

The train/test split is done by run, not by wafer. All wafers in a run share the same process condition, so splitting by wafer would leak the process condition into the test set. Splitting by run corresponds to measuring generalization to process conditions never seen during training.

## 3. Classification Models

Two models were implemented in numpy and compared.

The single-layer perceptron is trained with the pocket algorithm. Because the good region is an AND of intervals (a box) while the perceptron can draw only a single hyperplane, it is expected to be unable to solve the problem in principle. In practice, the number of misclassification corrections during training fails to decrease to the end — a non-convergence.

The MLP estimates P(good) with a hidden layer (16 neurons by default, configurable to multiple layers via the CLI) and a sigmoid output, and includes backpropagation, He initialization, and L2 regularization. To reflect the fab's cost asymmetry — a fail escape (FN) is more expensive than discarding a good wafer (FP) — the loss assigns a weight of 5 to fail samples (class-weighted BCE). This weighting alone raises fail recall at a 0.5 cutoff from 60.77% to 85.44%, and together with the inference-time cutoff provides two knobs for tuning the operating point.

## 4. Experiments and Results

The dataset is 11 runs × 20,000 = 220,000 wafers, trained on 9 runs and evaluated on 2.

### 4.1 Perceptron vs. MLP

| Metric (fail = positive) | Single-layer perceptron (pocket) | MLP (hidden 16) |
|---|---|---|
| Test accuracy | 79.86% | 83.82% |
| Precision | 34.63% | 55.22% |
| Recall | 4.95% | 85.44% |
| F1 | 8.66% | 67.08% |

Since a baseline that labels every wafer as good scores 80.70% accuracy, accuracy is not a meaningful criterion on this imbalanced data. The perceptron misses 95% of fails and does not even reach the baseline accuracy. As the number of features grows to five and the spec box becomes five-dimensional, the structural limitation — a single hyperplane cannot express "only the middle interval is good" — becomes even more pronounced.

![decision boundary](docs/decision_boundary.png)

In the figure above, the decision boundary is projected onto the Vth–Oxide plane (the remaining features fixed at their medians). The perceptron merely cuts the plane diagonally and thus labels all fails on one side as good, whereas the MLP has learned a closed region close to the actual spec box (dashed line).

### 4.2 Operating-point analysis

Because the MLP outputs a probability, the operating point can be selected via the decision cutoff. In settings where fail escapes are critical, the cutoff is raised to prioritize recall.

| cutoff | precision (fail) | recall (fail) | F1 |
|---|---|---|---|
| 0.1 | 79.94% | 52.60% | 63.45% |
| 0.5 | 55.22% | 85.44% | 67.08% |
| 0.9 | 33.67% | 98.11% | 50.13% |

Discriminative power across all thresholds — rather than at a specific cutoff — was evaluated with ROC and PR curves (`ml/visualize_metrics.py`). Since the perceptron has no probability output, its signed distance to the boundary was used as the score.

![model evaluation](docs/model_evaluation.png)

| Metric | Single-layer perceptron | MLP |
|---|---|---|
| ROC AUC | 0.386 | 0.925 |
| PR AP (fail) | 0.187 | 0.774 |

The perceptron's ROC AUC is below that of a random classifier (0.5). Because fails are scattered outside every face of the five-dimensional box, a single linear score cannot even rank the fails — pushing one end of the score axis toward "fail" leaves the fails at the other end missed.

### 4.3 Derived features: where expressive power lives

Since the spec limits are product requirements known to the fab, one can compute directly, as a feature, how far each measurement lies from its nearest spec boundary (a margin, normalized by the spec half-width) (`ml/features.py`, `--margins`). Adding the minimum of these margins, MinMargin, flattens the box rule "good iff every margin ≥ 0" into a linear cut along a single axis.

| Model | Features | recall (fail) | F1 |
|---|---|---|---|
| Perceptron | 5 raw | 4.95% | 8.66% |
| Perceptron | + 6 margins | 74.57% | 69.73% |
| MLP | 5 raw | 85.44% | 67.08% |
| MLP | + 6 margins | 87.34% | 68.71% |

The perceptron, which could not solve the box in principle, recovers to near the MLP's level. Inspecting the learned weights, it assigns the largest weight (+0.149) to MinMargin on its own. Encoding the geometry of the rule into the features leaves the model with only the judgment to make. The MLP, by contrast, gains little because its hidden layer can already express the box. Meanwhile neither reaches 100%, because the margins are computed from measured values while the labels are based on true values, so the ambiguity near the boundary cannot be removed by feature design.

### 4.4 sim2real: are data from a physics-mismatched simulator useful?

Up to Section 4.3, training and evaluation took place inside the same simulator; agreement is trivially expected because the simulator *is* reality. The genuinely meaningful question is whether the data are useful even when the simulator's physics differs from reality.

To this end, a second world playing the role of a "real fab" was defined. Its physical coupling coefficients differ from the digital twin's (and the twin does not know this difference), and its tester noise is about 40% larger (`REAL_FAB_PHYSICS` / `REAL_FAB_NOISE`, generated with `main.py --real`). The spec limits are product requirements and are shared between the two worlds.

Three strategies were compared: (i) scratch — train only on the affordable real data; (ii) zero-shot — train only on synthetic data and deploy as-is; (iii) fine-tune — pretrain on synthetic data, then fine-tune on a small amount of real data. Evaluation is fail-class F1 on two held-out runs (40,000 wafers) of the real fab, averaged over three seeds while varying the real-data budget (`ml/sim2real.py`).

| Strategy | 250 real | 1,000 | 4,000 | 16,000 |
|---|---|---|---|---|
| scratch (real data only) | 57.44% | 69.88% | 76.15% | 77.45% |
| fine-tune (synthetic pretrain + real) | 78.61% | 78.64% | 79.37% | 79.27% |

Reference lines: zero-shot F1 77.13%, oracle (all 20,000 real wafers) 76.56%.

![sim2real](docs/sim2real.png)

There are three observations.

First, a model with synthetic pretraining plus 250 real wafers outperforms a model trained from scratch on 20,000 real wafers (78.6% > 76.6%). The real-data requirement is effectively cut by about 80×.

Second, the cause of this advantage is not the quantity of data but the diversity of process conditions. The 20,000-wafer real pool comes from a single process condition (one run), whereas the twin's synthetic data span 12 conditions. No matter how many wafers of the same condition a model sees, it only learns the world near that one condition. This agrees with the fact that, in a real fab too, the bottleneck of data collection is condition coverage, not wafer count.

Third, fine-tune beats zero-shot across the whole range by 1.5–2.2 points. That is, the loss incurred by the physics mismatch is recovered by adapting on a small amount of real data.

Taken together, the value of synthetic data does not depend on the physical perfection of the simulator. This experiment set the twin's physics deliberately wrong and still showed that its data replace most of the real data. This measures robustness to a broken physical consistency, rather than verifying the physical consistency of the synthetic data.

## 5. Limitations and Future Work

The biggest limitation is that the "real fab" in this experiment is itself a simulator. Since the transfer is to a second virtual world with different physical constants, this is strictly a sim-to-sim experiment, and no validation against measured process data has been done. The inter-world difference is also modeled only as shifts in coupling coefficients and noise magnitude, whereas the real simulator-to-reality gap may take a more structural form, such as missing variables or non-stationarity.

The following are considered as future work.

- Validation against measured data (e.g., a public wafer-map dataset)
- A comparison of which is more data-efficient: calibrating the simulator's physical constants from a small amount of real data, versus fine-tuning the model
- Reporting metric variance (confidence intervals) from multi-seed training
- A comparison of a PyTorch reimplementation against the numpy implementation

## Appendix A. Repository Structure

```
virtual-fab/
├── main.py                     # simulation entry point
├── simulation/                 # the virtual fab (data generation)
│   ├── config.py               # process spec, sampling ranges, measurement noise, physics constants
│   ├── config_sampler.py       # samples a process condition per run (records the seed)
│   ├── wafer_generate.py       # generates wafer physical quantities + measurement noise
│   ├── wafer_analysis.py       # spec judgment → yield
│   ├── defect_analysis.py      # aggregation by fail cause
│   ├── visualization.py        # distribution / Pareto charts
│   ├── correlation_analysis.py # correlation analysis / scatter plots
│   ├── run_logger.py           # saves run_info.json + wafers.csv.gz
│   └── run_manage.py           # run-folder management (run_001, run_002, ...)
├── ml/                         # classifiers (numpy implementation)
│   ├── dataset.py              # run-aggregating loader, run-based train/test split
│   ├── perceptron.py           # single-layer perceptron + pocket algorithm
│   ├── mlp.py                  # MLP + backprop + weighted loss + L2 + CLI
│   ├── features.py             # derived features (margin to the spec boundary)
│   ├── metrics.py              # shared metrics (confusion matrix / PRF1, ROC·PR)
│   ├── judge.py                # judge a new run with the saved model (the EDS step)
│   ├── sim2real.py             # synthetic-pretrain vs real-data-budget experiment
│   ├── visualize_boundary.py   # decision-boundary visualization
│   └── visualize_metrics.py    # ROC·PR·training-curve·confusion-matrix visualization
└── graph/                      # run artifacts (gitignored)
    ├── run_XXX/                # per-run graphs, run_info.json, wafers.csv.gz
    └── real/run_XXX/           # "real fab" runs (main.py --real)
```

## Appendix B. Installation and Running

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Full pipeline (data generation → training → judgment):

```bash
python3 run_all.py                 # generate 10 runs → train & save the MLP → judge a new lot
python3 run_all.py --runs 20       # change the number of generated runs
python3 run_all.py --cutoff 0.9    # judge at a recall-first operating point
```

Step by step:

```bash
python3 main.py                  # generate data (accumulates 20k wafers under a new condition each run)
python3 ml/dataset.py            # accumulated-data status
python3 ml/perceptron.py         # train/evaluate the perceptron (--margins: derived features)
python3 ml/mlp.py                # train/evaluate the MLP, saves graph/ml/mlp_model.npz
python3 ml/judge.py run_011      # judge a new run with the saved model (cutoff configurable)
python3 ml/visualize_boundary.py # decision-boundary figure
python3 ml/visualize_metrics.py  # ROC·PR·training-curve·confusion-matrix figure
python3 main.py --real           # generate a "real fab" run (for sim2real)
python3 ml/sim2real.py           # the sim2real experiment
```
