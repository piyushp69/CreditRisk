"""Cached loaders for app/assets (built by app/build_assets.py) and the LightGBM model."""
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ASSETS = APP_DIR / "assets"
MODEL_FILE = APP_DIR / "models" / "lgbm_model.txt"

MODEL_ORDER = ["Logistic regression", "Decision tree", "Random forest", "XGBoost", "LightGBM"]
SHORT = {"Logistic regression": "logreg", "Decision tree": "dt", "Random forest": "rf", "XGBoost": "xgb",
         "LightGBM": "lgbm"}


def missing_assets():
    needed = ["facts.json", "metrics.csv", "predictor.json", "holdout_predictions.parquet"]
    return [n for n in needed if not (ASSETS / n).exists()] + ([] if MODEL_FILE.exists() else ["models/lgbm_model.txt"])


@st.cache_data(show_spinner=False)
def facts():
    return json.loads((ASSETS / "facts.json").read_text())


@st.cache_data(show_spinner=False)
def csv(name):
    return pd.read_csv(ASSETS / name)


@st.cache_data(show_spinner=False)
def parquet(name):
    return pd.read_parquet(ASSETS / name)


@st.cache_data(show_spinner=False)
def predictor_meta():
    return json.loads((ASSETS / "predictor.json").read_text())


@st.cache_resource(show_spinner="Loading the LightGBM model…")
def booster():
    return lgb.Booster(model_file=str(MODEL_FILE))


def figure(name):
    return ASSETS / "figures" / name


@st.cache_data(show_spinner=False)
def metrics_at(threshold_type):
    """Metrics table for one threshold type ('0.5' or 'cost-optimal'), indexed by model in MODEL_ORDER."""
    m = csv("metrics.csv")
    return m[m["threshold_type"].astype(str) == threshold_type].set_index("model").loc[MODEL_ORDER]


@st.cache_data(show_spinner=False)
def eda_rates(variable):
    d = csv("eda_default_rates.csv")
    return d[d["variable"] == variable].reset_index(drop=True)


def best_model():
    """Model with the highest holdout AUC among the five (all ranking metrics agree, see the comparison page)."""
    return facts()["models"]["best_by_auc"]
