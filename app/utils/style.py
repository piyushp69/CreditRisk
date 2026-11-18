"""Colours and number formats. Every model has one fixed colour on every page (validated palette order; the dark
column holds the dark-surface steps of the same hues)."""
import streamlit as st

from .loaders import MODEL_ORDER

MODEL_COLORS_LIGHT = {"Logistic regression": "#2a78d6", "Decision tree": "#e87ba4", "Random forest": "#eda100",
                      "XGBoost": "#1baf7a", "LightGBM": "#eb6834"}
MODEL_COLORS_DARK = {"Logistic regression": "#3987e5", "Decision tree": "#d55181", "Random forest": "#c98500",
                     "XGBoost": "#199e70", "LightGBM": "#d95926"}
# Secondary encoding for overlapping lines (identity never by colour alone).
MODEL_DASH = {"Logistic regression": "solid", "Decision tree": "dot", "Random forest": "dash", "XGBoost": "dashdot",
              "LightGBM": "solid"}
MODEL_SYMBOL = {"Logistic regression": "circle", "Decision tree": "diamond", "Random forest": "square",
                "XGBoost": "triangle-up", "LightGBM": "star"}

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 - no theme information (e.g. tests): light
        return False


def model_colors():
    return MODEL_COLORS_DARK if is_dark() else MODEL_COLORS_LIGHT


def series(i):
    """Categorical slot i (fixed order) for non-model series."""
    light = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    dark = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
    return (dark if is_dark() else light)[i]


def muted():
    return "#898781"


def diverging_scale():
    """Blue <-> red with a neutral grey midpoint."""
    mid = "#383835" if is_dark() else "#f0efec"
    return [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, mid], [0.75, "#ec835a"], [1.0, "#b8322f"]]


def sequential_scale():
    steps = SEQ_BLUE
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


# ---------------------------------------------------------------- number formats

def pct(x, d=2):
    return f"{x:.{d}%}"


def pp(x, d=2):
    """Percentage points from a difference of shares."""
    return f"{x * 100:+.{d}f} pp"


def num(x, d=0):
    return f"{x:,.{d}f}"


def f4(x):
    return f"{x:.4f}"


def secs(x):
    return f"{x:,.1f} s"


METRIC_LABELS = {"Accuracy": "Accuracy", "Precision": "Precision", "Recall": "Recall", "F1": "F1",
                 "LogLoss": "Log loss", "AUC": "AUC-ROC", "Gini": "Gini", "PR_AUC": "PR-AUC", "KS": "KS"}
HIGHER_IS_BETTER = {m: m != "LogLoss" for m in METRIC_LABELS}
THRESHOLD_FREE = ["AUC", "Gini", "PR_AUC", "KS", "LogLoss"]
THRESHOLD_BASED = ["Accuracy", "Precision", "Recall", "F1"]


def order_models(index):
    return [m for m in MODEL_ORDER if m in index]
