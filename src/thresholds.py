"""Classification metrics across decision thresholds (numpy only, so the Streamlit site can import it too).

Rule used everywhere in the project: an applicant is predicted to default (rejected) when score >= threshold.
"""
import numpy as np
import pandas as pd

CURVE_THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def confusion_at(y_true, y_prob, threshold):
    """(TN, FP, FN, TP) at one threshold."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_prob) >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return len(y) - tp - fp - fn, fp, fn, tp


def metrics_by_threshold(y_true, y_prob, thresholds=CURVE_THRESHOLDS):
    """Accuracy, precision, recall and F1 of the default class, and the confusion counts, at every threshold.
    Precision (and F1) are 0 when nothing is flagged, as in sklearn with zero_division=0."""
    rows = []
    for t in thresholds:
        tn, fp, fn, tp = confusion_at(y_true, y_prob, t)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(t), "Accuracy": (tp + tn) / (tn + fp + fn + tp), "Precision": precision,
                     "Recall": recall, "F1": f1, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                     "flagged_share": (tp + fp) / (tn + fp + fn + tp)})
    return pd.DataFrame(rows)
