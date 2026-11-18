"""SHAP explanations and plain-English reason codes for individual decisions.

`shap` is imported only inside `shap_explanation`, so the reason-code helpers can be used (e.g. by the Streamlit
site, which takes TreeSHAP values from LightGBM's own `pred_contrib`) without installing shap.
"""
import numpy as np
import pandas as pd


def shap_explanation(model, X):
    """TreeSHAP values for a fitted LightGBM classifier as a shap.Explanation (units: log-odds of default)."""
    import shap

    explanation = shap.TreeExplainer(model)(X)
    if explanation.values.ndim == 3:  # some shap versions return one slice per class
        explanation = explanation[:, :, 1]
    return explanation


def numeric_view(X):
    """Copy of X with categorical columns replaced by their integer codes (NaN for missing), for SHAP colouring."""
    X = X.copy()
    for col in X.select_dtypes("category").columns:
        X[col] = X[col].cat.codes.replace(-1, np.nan).astype("float64")
    return X


def mean_abs_shap(explanation, feature_names=None):
    names = feature_names or explanation.feature_names
    return pd.Series(np.abs(explanation.values).mean(axis=0), index=names).sort_values(ascending=False)


def _years(days):
    return -days / 365


def _yes_no(v):
    return "yes" if v == 1 else "no"


# feature -> function(value) returning a plain-English statement of the applicant's value.
# Reason codes list the features that push the predicted risk UP, so each template states the fact only.
TEMPLATES = {
    "EXT_SOURCE_1": lambda v: f"External credit score 1 is {v:.2f} (scale 0-1)",
    "EXT_SOURCE_2": lambda v: f"External credit score 2 is {v:.2f} (scale 0-1)",
    "EXT_SOURCE_3": lambda v: f"External credit score 3 is {v:.2f} (scale 0-1)",
    "EXT_SOURCE_MEAN": lambda v: f"Average external credit score is {v:.2f} (scale 0-1)",
    "EXT_SOURCE_STD": lambda v: f"External credit scores disagree (spread {v:.2f})",
    "ANNUITY_INCOME_RATIO": lambda v: f"Loan annuity is {v:.0%} of income",
    "CREDIT_INCOME_RATIO": lambda v: f"Loan amount is {v:.1f}x annual income",
    "CREDIT_TERM": lambda v: f"Annuity is {v:.1%} of the loan amount",
    "GOODS_CREDIT_RATIO": lambda v: f"Goods price covers {v:.0%} of the loan amount",
    "INCOME_PER_FAMILY_MEMBER": lambda v: f"Income per family member is {v:,.0f}",
    "DAYS_BIRTH": lambda v: f"Applicant is {_years(v):.0f} years old",
    "AGE_YEARS": lambda v: f"Applicant is {v:.0f} years old",
    "DAYS_EMPLOYED": lambda v: f"In current job for {_years(v):.1f} years",
    "YEARS_EMPLOYED": lambda v: f"In current job for {v:.1f} years",
    "EMPLOYED_TO_AGE": lambda v: f"Current job covers {v:.0%} of the applicant's life",
    "DAYS_ID_PUBLISH": lambda v: f"Identity document issued {_years(v):.1f} years before applying",
    "DAYS_REGISTRATION": lambda v: f"Registration changed {_years(v):.1f} years before applying",
    "DAYS_LAST_PHONE_CHANGE": lambda v: f"Phone number changed {-v:.0f} days before applying",
    "AMT_CREDIT": lambda v: f"Loan amount is {v:,.0f}",
    "AMT_ANNUITY": lambda v: f"Loan annuity is {v:,.0f}",
    "AMT_GOODS_PRICE": lambda v: f"Price of goods financed is {v:,.0f}",
    "AMT_INCOME_TOTAL": lambda v: f"Declared income is {v:,.0f}",
    "OWN_CAR_AGE": lambda v: f"Car is {v:.0f} years old",
    "REGION_RATING_CLIENT": lambda v: f"Home region risk rating is {v:.0f} (1 = best, 3 = worst)",
    "REGION_RATING_CLIENT_W_CITY": lambda v: f"Home region/city risk rating is {v:.0f} (1 = best, 3 = worst)",
    "DAYS_EMPLOYED_ANOM": lambda v: "No current employment recorded" if v == 1 else "Has current employment",
    "INST_LATE_SHARE": lambda v: f"Paid {v:.0%} of previous instalments late",
    "INST_DAYS_LATE_MEAN": lambda v: f"Previous instalments were paid {v:.1f} days late on average",
    "INST_DAYS_LATE_MAX": lambda v: f"Longest delay on a previous instalment was {v:.0f} days",
    "INST_PAYMENT_RATIO_MEAN": lambda v: f"Paid on average {v:.0%} of each previous instalment",
    "INST_PAYMENT_RATIO_MIN": lambda v: f"Lowest share of a previous instalment paid was {v:.0%}",
    "INST_PARTIAL_PAYMENT_SHARE": lambda v: f"{v:.0%} of previous instalments were paid in several parts",
    "INST_UNDERPAID_SHARE": lambda v: f"{v:.0%} of previous instalments were underpaid",
    "INST_COUNT": lambda v: f"{v:.0f} previous instalments on record",
    "PREV_APP_COUNT": lambda v: f"{v:.0f} previous applications with Home Credit",
    "PREV_REFUSED_SHARE": lambda v: f"{v:.0%} of previous applications were refused",
    "PREV_REFUSED_COUNT": lambda v: f"{v:.0f} previous applications were refused",
    "PREV_APPROVED_SHARE": lambda v: f"{v:.0%} of previous applications were approved",
    "PREV_APPROVED_COUNT": lambda v: f"{v:.0f} previous applications were approved",
    "PREV_DAYS_SINCE_LAST_DECISION": lambda v: f"Last previous application was decided {v:.0f} days ago",
    "PREV_APP_CREDIT_RATIO_MEAN": lambda v: f"Previously requested {v:.0%} of the credit granted, on average",
    "PREV_AMT_APPLICATION_MEAN": lambda v: f"Average previously requested amount is {v:,.0f}",
    "PREV_AMT_CREDIT_MEAN": lambda v: f"Average previously granted credit is {v:,.0f}",
    "BUR_LOAN_COUNT": lambda v: f"{v:.0f} credits reported to the Credit Bureau",
    "BUR_ACTIVE_COUNT": lambda v: f"{v:.0f} active credits at the Credit Bureau",
    "BUR_ACTIVE_SHARE": lambda v: f"{v:.0%} of Credit Bureau credits are still active",
    "BUR_DAYS_CREDIT_MEAN": lambda v: f"Credit Bureau credits were opened {-v:.0f} days ago on average",
    "BUR_DEBT_CREDIT_RATIO": lambda v: f"Outstanding bureau debt is {v:.0%} of bureau credit",
    "BUR_DEBT_SUM": lambda v: f"Total outstanding bureau debt is {v:,.0f}",
    "BUR_CREDIT_SUM_SUM": lambda v: f"Total bureau credit amount is {v:,.0f}",
    "BUR_CREDIT_SUM_MEAN": lambda v: f"Average bureau credit amount is {v:,.0f}",
    "BUR_OVERDUE_SUM": lambda v: f"Currently overdue bureau amount is {v:,.0f}",
    "BUR_MAX_DAY_OVERDUE": lambda v: f"Up to {v:.0f} days overdue on a bureau credit",
    "HAS_BUREAU_HISTORY": lambda v: "No Credit Bureau history" if v == 0 else "Has Credit Bureau history",
    "HAS_PREV_APPLICATION": lambda v: "No previous Home Credit application" if v == 0 else "Has previous applications",
    "HAS_INSTALLMENT_HISTORY": lambda v: "No previous instalment history" if v == 0 else "Has instalment history",
    "FLAG_OWN_CAR": lambda v: f"Owns a car: {v}",
    "FLAG_OWN_REALTY": lambda v: f"Owns real estate: {v}",
    "FLAG_DOCUMENT_3": lambda v: f"Provided document 3: {_yes_no(v)}",
    "CODE_GENDER": lambda v: f"Gender recorded as {v}",
    "NAME_EDUCATION_TYPE": lambda v: f"Education: {v}",
    "NAME_INCOME_TYPE": lambda v: f"Income type: {v}",
    "NAME_FAMILY_STATUS": lambda v: f"Family status: {v}",
    "NAME_CONTRACT_TYPE": lambda v: f"Contract type: {v}",
    "OCCUPATION_TYPE": lambda v: f"Occupation: {v}",
    "ORGANIZATION_TYPE": lambda v: f"Employer type: {v}",
}

MISSING_TEXT = {
    "EXT_SOURCE_1": "External credit score 1 is not available",
    "EXT_SOURCE_2": "External credit score 2 is not available",
    "EXT_SOURCE_3": "External credit score 3 is not available",
    "DAYS_EMPLOYED": "No current employment recorded",
    "YEARS_EMPLOYED": "No current employment recorded",
    "EMPLOYED_TO_AGE": "No current employment recorded",
    "OCCUPATION_TYPE": "Occupation not provided",
    "OWN_CAR_AGE": "No car age recorded (likely no car)",
}


def describe_feature(feature, value):
    """Plain-English statement for one feature value."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        if feature in MISSING_TEXT:
            return MISSING_TEXT[feature]
        if feature.startswith(("BUR_", "PREV_", "INST_")):
            return f"No data for {feature} (no matching history)"
        return f"{feature} is missing"
    template = TEMPLATES.get(feature)
    return template(value) if template else f"{feature} = {value}"


def reason_codes(shap_row, x_row, top_n=3):
    """Top-n features that increase this applicant's predicted risk, as (feature, SHAP, statement) rows."""
    contrib = pd.Series(shap_row, index=x_row.index)
    top = contrib[contrib > 0].sort_values(ascending=False).head(top_n)
    return pd.DataFrame({
        "feature": top.index,
        "shap_log_odds": top.values,
        "reason": [describe_feature(f, x_row[f]) for f in top.index],
    })
