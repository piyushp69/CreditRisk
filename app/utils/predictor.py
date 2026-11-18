"""Live loan-risk predictor: form inputs -> the 161 model features (built with the project's own src/ code) -> score
from the final LightGBM -> TreeSHAP reasons (LightGBM's own `pred_contrib`, in log-odds).

No Streamlit here, so app/build_assets.py can check that an unchanged example applicant reproduces its holdout score.
"""
import math

import numpy as np
import pandas as pd

from src.cleaning import DAYS_EMPLOYED_ANOMALY, clean_application
from src.explain import describe_feature
from src.features import add_application_ratios

EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
NOT_EMPLOYED_TYPES = ("Pensioner", "Unemployed")

# (key, label, kind, help). kind: "number" fields are compared after rounding to `decimals`.
FORM_FIELDS = [
    ("age", "Age (years)", 1, "Applicant's age on the application date"),
    ("income", "Annual income", 0, "Declared total annual income (data's own currency)"),
    ("credit", "Loan amount", 0, "Credit amount requested"),
    ("annuity", "Loan annuity (yearly repayment)", 0, "Yearly repayment amount of the loan"),
    ("goods_price", "Price of the goods financed", 0, "For consumer loans, the price of the goods bought"),
    ("years_employed", "Years in current job", 1, "0 for pensioners / unemployed applicants"),
    ("ext_source_mean", "Average external credit score (0–1)", 3, "Mean of the external scores EXT_SOURCE_1–3"),
    ("prev_refused", "Previous refused applications", 0, "Earlier Home Credit applications that were refused"),
    ("late_share", "Share of previous instalments paid late (0–1)", 3, "From earlier Home Credit loans"),
]
FORM_CATEGORIES = [("education", "NAME_EDUCATION_TYPE", "Education"),
                   ("income_type", "NAME_INCOME_TYPE", "Income type"),
                   ("contract_type", "NAME_CONTRACT_TYPE", "Contract type"),
                   ("family_status", "NAME_FAMILY_STATUS", "Family status")]
DECIMALS = {key: dec for key, _, dec, _ in FORM_FIELDS}


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def form_values(row):
    """The form inputs that describe `row` (a dict of model features), rounded as the form shows them."""
    years = 0.0 if row["DAYS_EMPLOYED_ANOM"] == 1 or _num(row["YEARS_EMPLOYED"]) is None else row["YEARS_EMPLOYED"]
    values = {
        "age": row["AGE_YEARS"], "income": row["AMT_INCOME_TOTAL"], "credit": row["AMT_CREDIT"],
        "annuity": row["AMT_ANNUITY"], "goods_price": row["AMT_GOODS_PRICE"], "years_employed": years,
        "ext_source_mean": row["EXT_SOURCE_MEAN"], "prev_refused": row["PREV_REFUSED_COUNT"],
        "late_share": row["INST_LATE_SHARE"] if _num(row["INST_LATE_SHARE"]) is not None else 0.0,
    }
    out = {k: round(float(v), DECIMALS[k]) if _num(v) is not None else 0.0 for k, v in values.items()}
    out.update({key: str(row[col]) for key, col, _ in FORM_CATEGORIES})
    return out


def _changed(inputs, shown, key):
    return round(float(inputs[key]), DECIMALS[key]) != shown[key] if key in DECIMALS else inputs[key] != shown[key]


def apply_inputs(base, inputs):
    """Copy of `base` (dict of model features) with the changed form inputs applied to the raw application columns
    and the side-table summaries. Unchanged inputs keep the base applicant's exact values."""
    row = dict(base)
    shown = form_values(base)
    days_employed = DAYS_EMPLOYED_ANOMALY if base["DAYS_EMPLOYED_ANOM"] == 1 else base["DAYS_EMPLOYED"]

    if _changed(inputs, shown, "age"):
        row["DAYS_BIRTH"] = -round(float(inputs["age"]) * 365)
    for key, col in [("income", "AMT_INCOME_TOTAL"), ("credit", "AMT_CREDIT"), ("annuity", "AMT_ANNUITY"),
                     ("goods_price", "AMT_GOODS_PRICE")]:
        if _changed(inputs, shown, key):
            row[col] = float(np.float32(inputs[key]))
    for key, col, _ in FORM_CATEGORIES:
        if _changed(inputs, shown, key):
            row[col] = inputs[key]
    if _changed(inputs, shown, "years_employed") or _changed(inputs, shown, "income_type"):
        years = float(inputs["years_employed"])
        days_employed = (DAYS_EMPLOYED_ANOMALY if years == 0 and inputs["income_type"] in NOT_EMPLOYED_TYPES
                         else -round(years * 365))
    if _changed(inputs, shown, "ext_source_mean"):
        target = float(inputs["ext_source_mean"])
        available = [c for c in EXT_COLS if _num(base[c]) is not None]
        if available:  # shift the available scores so their mean matches the input; their spread is kept
            delta = target - float(base["EXT_SOURCE_MEAN"])
            for c in available:
                row[c] = float(np.float32(min(1.0, max(0.0, base[c] + delta))))
        else:
            row["EXT_SOURCE_2"] = float(np.float32(target))
    if _changed(inputs, shown, "prev_refused"):
        refused = int(inputs["prev_refused"])
        apps = max(int(base["PREV_APP_COUNT"]), refused)
        approved = min(int(base["PREV_APPROVED_COUNT"]), apps - refused)
        row.update({"PREV_REFUSED_COUNT": refused, "PREV_APP_COUNT": apps, "PREV_APPROVED_COUNT": approved,
                    "PREV_REFUSED_SHARE": refused / apps if apps else np.nan,
                    "PREV_APPROVED_SHARE": approved / apps if apps else np.nan,
                    "HAS_PREV_APPLICATION": int(apps > 0)})
    if _changed(inputs, shown, "late_share"):
        row["INST_LATE_SHARE"] = float(np.float32(inputs["late_share"]))
    row["DAYS_EMPLOYED"] = days_employed
    return row


def build_features(base, inputs, features, categories):
    """One-row model input: raw columns -> src.cleaning.clean_application -> src.features.add_application_ratios."""
    row = apply_inputs(base, inputs)
    df = pd.DataFrame([{k: v for k, v in row.items() if k not in categories}])
    for col, cats in categories.items():
        value = row.get(col)
        df[col] = pd.Categorical([None if value is None or value == "nan" else value], categories=cats)
    df = clean_application(df)            # DAYS_EMPLOYED anomaly -> NaN + flag, AGE_YEARS, YEARS_EMPLOYED
    df = add_application_ratios(df)       # affordability ratios, EXT_SOURCE mean / spread
    for col, cats in categories.items():  # cleaning may drop unused levels; restore the training categories
        df[col] = df[col].astype(str).where(df[col].notna()).astype(pd.CategoricalDtype(cats))
    return df[features]


def score(booster, X):
    """Risk score (the class-weighted model's output; not a calibrated probability) and TreeSHAP contributions."""
    prob = float(booster.predict(X)[0])
    contrib = booster.predict(X, pred_contrib=True)[0]
    return prob, pd.Series(contrib[:-1], index=X.columns), float(contrib[-1])


def reasons(contrib, x_row, top_n=5):
    """The top_n features that move this applicant's score most, with plain-English statements (src.explain)."""
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    return pd.DataFrame({
        "feature": top.index,
        "shap_log_odds": top.to_numpy(),
        "effect": np.where(top.to_numpy() > 0, "raises risk", "lowers risk"),
        "reason": [describe_feature(f, x_row[f]) for f in top.index],
    })
