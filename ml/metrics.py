import numpy as np

# Shared metric helpers for both models.
#
# Convention across the project: label 1 = good, 0 = fail, and the
# class we actually care about catching is FAIL. So "positive" here
# means fail, and precision/recall/F1 are reported for the fail class.
#
# Curve helpers take a `good_score` (higher = more likely good): for
# the MLP that is P(good); for the perceptron it is the signed distance
# w . x + b. Only the ordering matters, so any monotonic good-score works.


def confusion(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 0) & (y_pred == 0)).sum())  # fail caught
    fn = int(((y_true == 0) & (y_pred == 1)).sum())  # fail missed
    fp = int(((y_true == 1) & (y_pred == 0)).sum())  # good rejected
    tn = int(((y_true == 1) & (y_pred == 1)).sum())  # good passed
    return tp, fp, fn, tn


# All the scalar metrics for one hard prediction, as a dict. Returning
# a dict (instead of printing) lets callers reuse the same computation
# for tables, sweeps, and figures without duplicating the arithmetic.
def binary_metrics(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    n = len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "baseline": float((np.asarray(y_true) == 1).mean()) if n else 0.0,
    }


# Human-readable dump of binary_metrics. Returns the dict too, so a
# caller can both print and keep the numbers.
def print_evaluation(y_true, y_pred):
    m = binary_metrics(y_true, y_pred)
    print(f"Accuracy : {m['accuracy'] * 100:.2f}%")
    print(f"(baseline 'all good' : {m['baseline'] * 100:.2f}%)")
    print()
    print("Confusion matrix (fail = positive)")
    print(f"  fail caught   (TP) : {m['tp']:6d}")
    print(f"  fail missed   (FN) : {m['fn']:6d}")
    print(f"  good rejected (FP) : {m['fp']:6d}")
    print(f"  good passed   (TN) : {m['tn']:6d}")
    print()
    print(f"Precision (fail) : {m['precision'] * 100:.2f}%")
    print(f"Recall    (fail) : {m['recall'] * 100:.2f}%")
    print(f"F1        (fail) : {m['f1'] * 100:.2f}%")
    return m


# Sort samples once by how fail-like they are (lowest good-score first),
# then read every operating point off cumulative counts -- O(n log n)
# instead of re-scanning the whole array per threshold.
def _cumulative_fail_counts(y_true, good_score):
    fail = (np.asarray(y_true) == 0).astype(int)
    order = np.argsort(np.asarray(good_score, dtype=float), kind="mergesort")
    fail_sorted = fail[order]
    tp = np.cumsum(fail_sorted)          # fails caught as we lower the bar
    fp = np.cumsum(1 - fail_sorted)      # goods wrongly flagged
    return tp, fp, int(fail.sum()), int((fail == 0).sum())


# ROC: false-positive rate (goods rejected) vs true-positive rate
# (fails caught) as the fail threshold sweeps. AUC via the trapezoid rule.
def roc_curve(y_true, good_score):
    tp, fp, P, N = _cumulative_fail_counts(y_true, good_score)
    tpr = np.concatenate([[0.0], tp / P]) if P else np.array([0.0, 0.0])
    fpr = np.concatenate([[0.0], fp / N]) if N else np.array([0.0, 0.0])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


# Precision-recall for the fail class. Average precision via trapezoid.
def pr_curve(y_true, good_score):
    tp, fp, P, _ = _cumulative_fail_counts(y_true, good_score)
    recall = np.concatenate([[0.0], tp / P]) if P else np.array([0.0, 0.0])
    precision = np.concatenate([[1.0], tp / (tp + fp)])
    return recall, precision, float(np.trapezoid(precision, recall))
