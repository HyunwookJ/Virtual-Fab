# Virtual Fab — Learning a Wafer Pass/Fail Classifier from Simulated Process Data

> **Research question:** Can cheaply generated synthetic wafer data replace expensive real wafers?
> **Summary:** To a large extent, yes. This document measures the extent quantitatively.

This project builds a simulator of a semiconductor process (a virtual fab) and uses
the data it generates to train a good/fail classifier for the EDS test step. The
simulator and the classifiers (perceptron, MLP, backpropagation) are implemented
**entirely in numpy**, without scikit-learn, PyTorch, or any other ML framework.
Implementing the internals directly, rather than relying on a framework, is itself
one of the goals of the project.

> 한국어판: [`README.ko.md`](README.ko.md)

---

## 0. Overview

**Components**

| | Content | Location |
|---|---|---|
| 1 | **Virtual fab** — a simulator that generates 20,000 wafers of process data per execution | `simulation/` |
| 2 | **Classifier** — classifies those wafers as good or failed (implemented directly in numpy) | `ml/` |

**Key results**

1. The good region is a **simultaneous satisfaction (AND)** of several conditions,
   so a single-layer perceptron — which can only draw one hyperplane — cannot solve
   the problem structurally. It missed 95% of the fails. An MLP with a hidden layer
   solves it. (→ [Section 4](#4-the-two-classification-models))
2. With **appropriately designed features**, however, the perceptron recovers to a
   practical level: fail recall rises from 4.95% to 74.57%. Expressive power resides
   not only in the model but also in the features.
   (→ [Section 5.4](#54-the-effect-of-derived-margin-features))
3. Even when the simulator's physics is set **deliberately different** from reality,
   a model pretrained on synthetic data remains effective. A model given a further
   **250 real wafers** outperformed one trained from scratch on **20,000 real
   wafers**. (→ [Section 6](#6-when-the-simulators-physics-differs-from-reality-sim2real))

**Suggested reading order:** if time is limited, Sections 1 → 3.3 → 6 convey the
core argument on their own.

---

## 1. Background and problem definition

In a typical machine learning exercise the data is already available. If more is
needed it can be collected, and poor-quality data can be discarded.

In a semiconductor fab this premise does not hold.

- **The unit cost of a wafer is high.** Obtaining one row of data requires
  physically passing a wafer through dozens of process steps and then measuring it.
- **Fail samples are especially scarce.** The higher the yield, the rarer the
  fails — yet fails are precisely what the model must learn.
- **Collecting data across varying process conditions is the most expensive part.**
  Deliberately shifting the temperature or settings of a production line costs far
  more than simply increasing the wafer count at a fixed condition.

This motivates the following approach: **generate synthetic wafers cheaply and in
volume using a simulator that approximates the process physics, and train the
classifier on that data.** In industry such a simulator is called a
**digital twin**.

This project verifies whether the approach actually holds, implementing everything
from the simulator to the model evaluation. Three questions are examined.

1. Can pass/fail judgment **be learned** from simulated data, and what model
   capacity does the structure of the problem require? (Sections 2–5)
2. Does the data remain valid even when the simulator's physics **disagrees with
   reality**? (Section 6)
3. If so, **to what extent can it replace** real data? (Section 6)

---

## 2. Background terminology

The semiconductor terms required to read this document are summarized below. ML
terms are explained where they first appear, and the full list is in
[Appendix C](#appendix-c-glossary).

| Term | Meaning |
|---|---|
| **Wafer** | The circular silicon substrate on which devices are formed. In this project, **one row of data = one wafer**. |
| **Fab** | A semiconductor factory (fabrication plant). |
| **EDS** | The test step that probes a finished wafer electrically and sorts good from failed. **This is the step the model is intended to replace.** |
| **Spec** | The range of values a product must satisfy — e.g. a threshold voltage between 0.67 V and 0.73 V. Specs are not physical laws but **product requirements**: chosen by people and already known to the fab. |
| **Yield** | The fraction of produced wafers that are good. |
| **Run** | In this project, **one execution of the simulator**: 20,000 wafers, all generated under the **same process condition**. This property becomes important in [Section 3.4](#34-run-based-traintest-splitting). |

---

## 3. Design of the virtual fab

> Code: `simulation/`

### 3.1 Wafer representation: five process parameters

Each execution of the simulator generates 20,000 wafers, and each wafer is
represented by five values.

| Parameter | Meaning | Spec range |
|---|---|---|
| `Vth[V]` | Threshold voltage — the voltage at which the transistor begins to turn on | 0.67 – 0.73 |
| `Oxide[nm]` | Gate oxide thickness | 97 – 103 |
| `Leakage[nA]` | Leakage current — current flowing while the device is off | ≤ 10 (upper limit only) |
| `CD[nm]` | Gate critical dimension | 43.5 – 46.5 |
| `Temp[C]` | Process temperature | 23 – 27 |

**A wafer is good only if all five parameters lie within their ranges**; a single
violation makes it a fail. The good condition is therefore an AND of intervals,
which geometrically amounts to asking whether a point lies inside a **box** in
5-D space. This box shape determines the entire argument of
[Section 4](#4-the-two-classification-models).

For each run the **center and spread** of every parameter are sampled at random
within fixed ranges (`PARAM_RANGES` in `simulation/config.py`), reproducing the
way a real line sits slightly differently from period to period. Different runs
therefore have different data distributions.

### 3.2 Physical correlations among parameters

Sampling the five values independently would produce random numbers rather than
process data. The parameters are therefore coupled according to device physics
(`PHYSICS`).

- **Thicker** oxide → threshold voltage **rises**, leakage **falls**
- **Narrower** gate (CD) → threshold voltage **falls**, leakage **rises**
  (short-channel effect)
- **Higher** temperature → leakage increases **exponentially**

This gives the data genuine correlation structure, and therefore something for the
model to learn.

### 3.3 Labels from true values, features from measurements

This is the most important design decision in the project.

- **Labels are determined from the true values.** The simulator knows each wafer's
  actual physical values, so good/fail is decided by whether those true values lie
  within spec.
- **The features given to the model are measurements** — the true value plus sensor
  noise, i.e. **the number a tester would actually report**. Only these are stored
  in the dataset.

The reason is that a real fab's test equipment reports nothing but measurements.
This design has the following consequence:

> If the true Vth is 0.7301, it exceeds the spec limit (0.73) and the wafer is a
> **fail**. If the measurement is recorded as 0.7295, however, a model observing
> only the data sees a **good** wafer.

That is, wafers near a spec boundary **cannot be classified from measurements
alone, even in principle**. The task therefore becomes an inherently probabilistic
classification problem in which no model can reach 100% accuracy — a property
introduced deliberately, so that the problem does not have a trivial solution.

The property is guaranteed by ordering: in `main.py` the label is judged from the
true values **first**, after which noise is added and the feature columns are
overwritten. Reversing that order destroys the property.

### 3.4 Run-based train/test splitting

Rather than shuffling wafers at random, whole runs are held out as the test set.

The reason lies in the property noted in
[Section 2](#2-background-terminology): all 20,000 wafers in a run are generated
under the same process condition. Splitting by wafer means that, for every test
wafer, effectively identical wafers were already observed during training — the
equivalent of seeing the exam material in advance, which inflates the measured
performance.

Splitting by run instead measures generalization: **can the model judge process
conditions it never observed during training?**
(`load_train_test_by_run()` in `ml/dataset.py`)

---

## 4. The two classification models

> Code: `ml/`

### 4.1 The single-layer perceptron

The **perceptron** is the simplest classifier. It multiplies each input by a
weight, sums the products, and assigns one class if the sum exceeds zero and the
other class otherwise. Geometrically, all it can do is **draw a single line (a
hyperplane in higher dimensions) and divide the space in two**.

Training uses the **pocket algorithm**. An ordinary perceptron whose data cannot be
separated by a single hyperplane never converges — the weights oscillate
indefinitely, and the weights at the moment training stops may happen to be poor.
The pocket algorithm keeps **a separate copy of the best-performing weights seen
during training** and ships those, rather than the final ones, as the model.
(`ml/perceptron.py`)

### 4.2 The structural limitation of the single-layer perceptron

Reducing the problem to one dimension makes the limitation clear. Considering Vth
alone:

```
fail ←|————— good —————|→ fail
    0.67              0.73
```

The good interval lies **in the middle** while fails are distributed on **both
sides**. The only decision rule a perceptron can express, however, is "good if
above this value" or "good if below this value." **The rule "only the middle
interval is good" cannot be represented by a single hyperplane.**

As the parameter count grows to five, the good region becomes a 5-D box and the
fails are distributed **outside every face, in every direction**, making the
limitation more pronounced still. The perceptron fails here not from insufficient
training but because the task is **structurally impossible** for it — and indeed
the number of misclassification corrections never decreases through training.

### 4.3 The MLP: closed regions from a hidden layer

An **MLP** (multi-layer perceptron) places a **hidden layer** between the input and
output layers (16 neurons by default). Each hidden neuron accounts for one
hyperplane and the output layer combines them, so the network can represent a
**closed region** — a box — formed from several hyperplanes. This is precisely what
the perceptron could not do.

The implementation includes the following (`ml/mlp.py`).

- **Backpropagation** — propagating the output error back through the preceding
  layers to compute the correction for each weight. Implemented directly.
- **He initialization** — scaling the initial weights to the size of the layer so
  that the signal neither vanishes nor explodes as it passes through.
- **L2 regularization** — penalizing excessively large weights to curb overfitting.
- **Sigmoid output** — mapping the output into the interval 0 to 1. This value is
  interpreted as **P(good), the probability that the wafer is good**. Producing a
  probability rather than a binary verdict is exploited in
  [Section 5.3](#53-the-decision-cutoff-and-the-operating-point).

### 4.4 Cost asymmetry and class weighting

From the perspective of the fab, the two kinds of misclassification carry different
costs.

- **Passing a failed wafer as good** → it reaches the customer. Critical.
- **Scrapping a good wafer as failed** → the loss is limited to one wafer.

Since roughly 80% of the data consists of good wafers, however, an unadjusted model
drifts toward answering "good" in most cases, because doing so is less often wrong
on average.

To correct this, the loss function counts **each failed wafer with the weight of
five good wafers** (`fail_weight=5`, class-weighted BCE). This weighting alone
raised fail recall from **60.77% to 85.44%**.

---

## 5. Experiments and results

The dataset comprises 11 runs × 20,000 = **220,000 wafers**, trained on 9 runs and
evaluated on the remaining 2 (40,000 wafers).

### 5.1 Choice of evaluation metric

On this data, **labeling every wafer as good already yields 80.70% accuracy**,
because approximately 80% of the test set is genuinely good. Since a model that has
learned nothing scores 80 out of 100, accuracy cannot serve as a criterion.

Evaluation therefore rests on **how well the fails are detected** (fail = positive).

| Metric | Definition | Example reading |
|---|---|---|
| **Precision** | Of the wafers judged to be fails, the fraction that genuinely failed | 34.63% = only about a third of the flagged wafers were true fails; the remainder were sound wafers that were scrapped |
| **Recall** | Of the wafers that genuinely failed, the fraction the model detected | 4.95% = 5 fails detected out of every 100, with **95 passed through** |
| **F1** | The harmonic mean of the two, used when a single figure is needed | Scoring well on only one of the two keeps it low |

### 5.2 Perceptron compared with MLP

| Metric (fail = positive) | Single-layer perceptron | MLP (hidden 16) |
|---|---|---|
| Test accuracy | 79.86% | 83.82% |
| Precision | 34.63% | 55.22% |
| Recall | **4.95%** | **85.44%** |
| F1 | 8.66% | 67.08% |

The perceptron missed 95% of the fails, and its accuracy does not even reach the
"label everything good" baseline of 80.70% — consistent with the prediction made in
[Section 4.2](#42-the-structural-limitation-of-the-single-layer-perceptron).

![decision boundary](docs/decision_boundary.png)

The figure above projects the decision boundary onto the Vth–Oxide plane (the
remaining three parameters fixed at their medians). The perceptron merely
**partitions the plane along a diagonal**, so every fail on one side is labeled
good, whereas the MLP has learned a **closed region** close to the actual spec box
(dashed line).

### 5.3 The decision cutoff and the operating point

Because the MLP outputs the probability P(good), the threshold above which a wafer
is passed can be chosen at inference time. That threshold is the cutoff, with a
default of 0.5, and the decision criterion it defines is called the
**operating point**.

**Raising** the cutoff makes the criterion stricter, increasing fail detection
(recall ↑) while also increasing the scrapping of sound wafers (precision ↓).

| cutoff | Precision (fail) | Recall (fail) | F1 |
|---|---|---|---|
| 0.1 | 79.94% | 52.60% | 63.45% |
| 0.5 | 55.22% | 85.44% | 67.08% |
| 0.9 | 33.67% | **98.11%** | 50.13% |

On a line where an escaped fail is critical, the cutoff can be set to 0.9 to
prioritize recall. The operating point is thus adjustable through two parameters:
`fail_weight` at training time and `cutoff` at inference time
(`python3 ml/judge.py run_011 0.9`).

Discriminative power **across all thresholds**, rather than at one cutoff, is
evaluated with ROC and PR curves. ROC AUC can be read as "the probability that,
given one random fail and one random good wafer, the model scores the fail as the
more fail-like of the two"; 0.5 corresponds to a random classifier.

![model evaluation](docs/model_evaluation.png)

| Metric | Single-layer perceptron | MLP |
|---|---|---|
| ROC AUC | **0.386** | 0.925 |
| PR AP (fail) | 0.187 | 0.774 |

The perceptron's ROC AUC falls **below that of a random classifier (0.5)**. Its
score increases in only one direction, whereas the fails are distributed **outside
every face** of the box. Pushing one end of the score axis toward "fail" causes the
fails at the opposite end to be ranked as the most good-like wafers of all. The
model fails even to rank the fails correctly.

### 5.4 The effect of derived margin features

Spec limits are not physical laws but **product requirements the company already
possesses**, so supplying them to the model does not constitute information
leakage. Accordingly, the distance of each measurement from its **nearest spec
boundary** is computed directly and added as a feature (a margin, normalized by the
spec width). A positive value indicates the measurement is inside spec, a negative
value outside.

Adding the **minimum** of those margins (`MinMargin`, the tightest of the five)
transforms the decision rule as follows.

> **Original rule:** all five values inside spec → good (a 5-D box)
> **Transformed rule:** `MinMargin ≥ 0` → good (**a linear cut along a single axis**)

A problem unsolvable with one hyperplane becomes one that a single hyperplane
solves. The results are as follows.

| Model | Features | Recall (fail) | F1 |
|---|---|---|---|
| Perceptron | 5 raw | 4.95% | 8.66% |
| Perceptron | **+ 6 margins** | **74.57%** | **69.73%** |
| MLP | 5 raw | 85.44% | 67.08% |
| MLP | + 6 margins | 87.34% | 68.71% |

The perceptron, structurally incapable of solving the problem before, recovers to
near the MLP's level. Inspecting its learned weights shows that it assigned the
largest weight (+0.149) to `MinMargin` on its own. Encoding the geometry of the
rule into the features leaves the model with only the judgment to make.

The MLP, by contrast, gains little, since its hidden layer can already express the
box and the additional information adds little utility. **Expressive power resides
both in the model and in the features, and strengthening one reduces the demand on
the other** — the conclusion of this section.

Neither model reaches 100%, however, because the margins are computed from
**measured** values while the labels are based on **true** values
([Section 3.3](#33-labels-from-true-values-features-from-measurements)). Ambiguity
near the boundary cannot be removed through feature design.

> To run: `python3 ml/perceptron.py --margins`, `python3 ml/mlp.py --margins`
> (A model trained with `--margins` records that fact in the model file, and
> `judge.py` applies the identical transform to fresh wafers automatically. The CSV
> continues to store only the five raw columns.)

---

## 6. When the simulator's physics differs from reality (sim2real)

### 6.1 A limitation of the experiments so far

Up to Section 5, training and evaluation both took place **inside the same
simulator**. In a world where the simulator *is* reality, good agreement is to be
expected, and practical validity remains unverified.

The question of real substance is therefore: **does the data remain valid when the
simulator's physics disagrees with reality?** Since no simulator reproduces the
physics of reality completely, failing to answer this question leaves the entire
approach without foundation.

### 6.2 Two worlds

To address this, a second world was defined to play the role of a "real fab."

| Item | Digital twin (virtual fab) | Real fab |
|---|---|---|
| Physics constants | `PHYSICS` | `REAL_FAB_PHYSICS` — **different** (e.g. oxide→Vth coupling 0.010 vs 0.013) |
| Measurement noise | `MEASUREMENT_NOISE` | `REAL_FAB_NOISE` — approximately **40% larger** (real testers are less precise) |
| Spec limits | shared | shared (product requirements, identical across both worlds) |
| Cost of data | free, 12 process conditions | expensive, budget-limited |
| Location | `graph/run_*` | `graph/real/run_*` (`main.py --real`) |

Notably, **the twin has no knowledge of this difference**. The synthetic data here
is not merely "data containing noise" but **a systematically biased distribution
generated from incorrect physical laws**. Whether it nonetheless remains useful is
the object of this experiment.

### 6.3 The three strategies compared

Three methods of building an EDS model for the real fab were compared.

| Strategy | Description | Analogy |
|---|---|---|
| **scratch** | Train from the beginning using only the affordable real data | Sitting the exam without past papers |
| **zero-shot** | Train on synthetic data alone and deploy without using any real data | Preparing only in advance, then sitting the exam |
| **fine-tune** | Pretrain on synthetic data, then train further on a small amount of real data | Preparing in advance, then adjusting on actual questions |

Evaluation uses fail-class F1 on **two held-out runs (40,000 wafers) of the real
fab**, averaged over 3 seeds while varying the real-data budget (`ml/sim2real.py`).

### 6.4 Results

| Strategy | 250 real | 1,000 | 4,000 | 16,000 |
|---|---|---|---|---|
| scratch (real data only) | 57.44% | 69.88% | 76.15% | 77.45% |
| **fine-tune (synthetic pretrain + real)** | **78.61%** | **78.64%** | **79.37%** | **79.27%** |

Reference lines: zero-shot **77.13%**, oracle (all 20,000 real wafers) **76.56%**.

![sim2real](docs/sim2real.png)

**First, a model combining 250 real wafers with synthetic pretraining outperformed
a model trained from scratch on 20,000 real wafers** (78.6% > 76.6%). The real-data
requirement is effectively reduced by a factor of about **80**.

**Second, the cause of this advantage is not the quantity of data but the diversity
of process conditions.** All 20,000 real wafers were generated under a **single
process condition** (one run). However many wafers of one condition a model
observes, it learns only the distribution near that condition, whereas the twin's
synthetic data spans **12 conditions**. By analogy, twelve schools' past papers
cover a broader examination scope than twenty thousand questions from a single
school. This is consistent with the fact that, in a real fab as well, the
bottleneck in data collection is **condition coverage** rather than wafer count.

**Third, fine-tune outperforms zero-shot across the entire range by 1.5–2.2
points.** The loss incurred by the physics mismatch is thus recovered by adapting
on a small amount of real data.

Taken together, **the value of synthetic data does not depend on the physical
completeness of the simulator.** This experiment deliberately set the twin's
physics differently and still showed that its data replaces most of the real data.

---

## 7. Conclusions and limitations

### 7.1 What this project claims

> The value of synthetic data does not derive from physical completeness. It
> persists even when the simulator disagrees with reality, and the loss caused by
> that mismatch can be recovered by fine-tuning on a small amount of real data.

### 7.2 What this project does not claim

- This experiment does not **validate** synthetic data against real physics. On the
  contrary, it sets the physics **deliberately differently** and examines whether
  usefulness is retained — a **robustness** experiment rather than a validation.
- Correcting the simulator itself from real data (back-solving its physics
  constants, i.e. **calibration**) is the opposite direction and a separate
  problem. Which is more efficient — adapting the model to reality, or adapting the
  simulator to reality — lies outside the scope of this experiment.

### 7.3 Limitations

The most significant limitation is that the "real fab" in this experiment is
**also a simulator**. The transfer is to a second virtual world differing only in
its physics constants, so this is strictly a sim-to-**sim** experiment, and no
validation against measured process data was performed. Furthermore, the difference
between the worlds is modeled only as shifts in **coupling constants and noise
magnitude**, whereas the actual simulator-to-reality gap may take a more structural
form, such as missing variables or drift over time.

### 7.4 Future work

- Validation against measured data (e.g. a public wafer-map dataset)
- A comparison of which is more data-efficient: calibrating the simulator's physics
  constants from a small amount of real data, or fine-tuning the model
- Reporting metric variance (confidence intervals) from multi-seed training
- A comparison of a PyTorch reimplementation against the numpy implementation

---

## Appendix A. Installation and running

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Full pipeline** (data generation → training → judgment):

```bash
python3 run_all.py                 # generate 10 runs → train & save the MLP → judge a fresh lot
python3 run_all.py --runs 20       # change the number of generated runs
python3 run_all.py --cutoff 0.9    # judge at a recall-first operating point
```

**Step by step** (run every command from the repository root):

```bash
python3 main.py                  # generate data (each call accumulates 20k wafers under a new condition)
python3 main.py --real           # generate a "real fab" run (for the Section 6 experiment)
python3 ml/dataset.py            # print the status of the accumulated data

python3 ml/perceptron.py         # train/evaluate the perceptron (--margins: derived features)
python3 ml/mlp.py                # train/evaluate the MLP → saves graph/ml/mlp_model.npz
python3 ml/mlp.py --hidden 32 16 --l2 1e-4 --epochs 50   # multi-layer configuration and hyperparameters
python3 ml/judge.py run_011      # judge a new run with the saved model (the EDS step)
python3 ml/judge.py run_011 0.9  # judge with a specified cutoff

python3 ml/sim2real.py           # the Section 6 experiment → graph/ml/sim2real.png
python3 ml/visualize_boundary.py # decision-boundary visualization
python3 ml/visualize_metrics.py  # ROC · PR · training-curve · confusion-matrix visualization
```

There is no separate test suite or linter; verification consists of running the
scripts and reading the metrics they print.

## Appendix B. Repository structure

```
virtual-fab/
├── main.py                     # simulation entry point
├── run_all.py                  # generation → training → judgment, in one command
├── simulation/                 # the virtual fab (data generation)
│   ├── config.py               # spec limits, sampling ranges, measurement noise, physics ★source of truth
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
│   ├── metrics.py              # shared metrics (confusion matrix / PRF1, ROC · PR)
│   ├── judge.py                # judge a new run with the saved model (the EDS step)
│   ├── sim2real.py             # synthetic-pretrain vs real-data-budget experiment
│   ├── visualize_boundary.py   # decision-boundary visualization
│   └── visualize_metrics.py    # ROC · PR · training-curve · confusion-matrix visualization
└── graph/                      # run artifacts (gitignored, regenerable)
    ├── run_XXX/                # per-run graphs, run_info.json, wafers.csv.gz
    └── real/run_XXX/           # "real fab" runs (main.py --real)
```

The two stages (`simulation/` and `ml/`) are connected **only through files on
disk**: the simulation writes `graph/run_XXX/wafers.csv.gz` and the ML stage reads
it back. The column names (`FEATURE_COLUMNS` / `LABEL_COLUMN` in `ml/dataset.py`)
constitute the contract between the two stages, so changing one side requires
changing the other.

## Appendix C. Glossary

| Term | Meaning |
|---|---|
| **Feature** | The values supplied to the model as input. Here, the five measured parameters (and optionally 6 margins) |
| **Label** | The correct answer. Here, good (1) / fail (0), judged from the true values |
| **Perceptron** | The simplest classifier: divides the space in two with a single hyperplane |
| **Pocket algorithm** | Retains the best-performing weights seen during training and uses them as the final model |
| **MLP** | A neural network with a hidden layer; can represent closed regions by combining hyperplanes |
| **Hidden layer** | The intermediate layer between input and output, where the model's expressive power arises |
| **Backpropagation** | Propagating the output error back through the layers to compute each weight's correction |
| **He initialization** | Scaling the initial weights to the layer size to prevent the signal vanishing or exploding |
| **L2 regularization** | Penalizing the magnitude of the weights to curb overfitting |
| **Sigmoid** | Maps any real number into the interval 0 to 1; used at the output for probabilistic interpretation |
| **Loss function** | A function quantifying how wrong the model is. Training is the process of minimizing it |
| **Class weight** | Counting one class's samples more heavily in the loss. Here, fails count 5× |
| **Cutoff** | The threshold converting a probability into a verdict: good if `P(good) ≥ cutoff` |
| **Operating point** | The precision/recall balance determined by the selected cutoff |
| **Precision / Recall / F1** | See [Section 5.1](#51-choice-of-evaluation-metric) |
| **ROC AUC** | Discriminative power across all thresholds expressed as a single figure. 0.5 = a random classifier |
| **Overfitting** | Fitting the training data well while failing to generalize to new data |
| **Digital twin** | A simulator built to mirror a real system |
| **sim2real** | The problem of applying what was learned in a simulator to reality |
| **Zero-shot** | Applying a model as-is, without using any data from the target domain |
| **Fine-tune** | Adapting a pretrained model by training it further on a small amount of target data |
