"""Cleaning for the application tables (application_train / application_test)."""
import pandas as pd

from .config import TARGET

# Placeholder Home Credit uses in DAYS_EMPLOYED (~1,000 years) for applicants without employment
# (mostly pensioners / unemployed).
DAYS_EMPLOYED_ANOMALY = 365243


def clean_application(df):
    """Return a cleaned copy of an application table.

    - DAYS_EMPLOYED == 365243 -> NaN, flagged in DAYS_EMPLOYED_ANOM (int8)
    - AGE_YEARS = -DAYS_BIRTH / 365, YEARS_EMPLOYED = -DAYS_EMPLOYED / 365
    - rows with CODE_GENDER == 'XNA' dropped
    Income outliers are NOT removed here; see income_outlier_report.
    """
    df = df.copy()
    anomalous = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
    df["DAYS_EMPLOYED_ANOM"] = anomalous.astype("int8")
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].astype("float32").where(~anomalous)

    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).astype("float32")
    df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365 + 0.0).astype("float32")  # + 0.0 avoids -0.0

    df = df[df["CODE_GENDER"] != "XNA"].reset_index(drop=True)
    if isinstance(df["CODE_GENDER"].dtype, pd.CategoricalDtype):
        df["CODE_GENDER"] = df["CODE_GENDER"].cat.remove_unused_categories()
    return df


def days_employed_anomaly_report(df, target=TARGET):
    """Count of DAYS_EMPLOYED == 365243 rows and default rate with vs without the anomaly (raw data)."""
    anomalous = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
    return pd.DataFrame({
        "applicants": [int(anomalous.sum()), int((~anomalous).sum())],
        "share": [anomalous.mean(), 1 - anomalous.mean()],
        "default_rate": [df.loc[anomalous, target].mean(), df.loc[~anomalous, target].mean()],
    }, index=["DAYS_EMPLOYED = 365243", "valid DAYS_EMPLOYED"])


def income_outlier_report(df, quantile=0.999, target=TARGET, top_n=10):
    """Return (threshold, n_above, default rate above threshold, top_n highest incomes) for AMT_INCOME_TOTAL.

    Reporting only - outliers are kept in the data.
    """
    income = df["AMT_INCOME_TOTAL"]
    threshold = float(income.quantile(quantile))
    above = income > threshold
    top = df.nlargest(top_n, "AMT_INCOME_TOTAL")[["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", target]]
    return threshold, int(above.sum()), float(df.loc[above, target].mean()), top
