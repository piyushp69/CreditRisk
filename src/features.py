"""Feature engineering: application ratios and side-table aggregates, one row per SK_ID_CURR.

Side tables are aggregated by SK_ID_CURR first and then left-joined onto the cleaned application
table. Every aggregate is a per-applicant summary of that applicant's own rows (no statistics are
learned across applicants), so building them before the train/holdout split does not leak.
"""
import pandas as pd

from .config import ID, PROCESSED_DIR

EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# Count columns that are 0 (not missing) for applicants without rows in the side table.
COUNT_COLUMNS = ["BUR_LOAN_COUNT", "BUR_ACTIVE_COUNT", "PREV_APP_COUNT", "PREV_APPROVED_COUNT",
                 "PREV_REFUSED_COUNT", "INST_COUNT"]


def safe_div(num, den):
    """num / den as float32; NaN where den is 0 or missing (never inf)."""
    den = den.astype("float64")
    return (num.astype("float64") / den.where(den != 0) + 0.0).astype("float32")  # + 0.0 turns -0.0 into 0.0


def add_application_ratios(df):
    """Add affordability / stability ratios and EXT_SOURCE summaries.

    Expects a table already passed through src.cleaning.clean_application (DAYS_EMPLOYED anomaly as NaN).
    """
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_TERM"] = safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["GOODS_CREDIT_RATIO"] = safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["EMPLOYED_TO_AGE"] = safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["INCOME_PER_FAMILY_MEMBER"] = safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])
    ext = df[EXT_SOURCES].astype("float64")
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1).astype("float32")  # mean of the available scores
    df["EXT_SOURCE_STD"] = ext.std(axis=1).astype("float32")    # NaN when fewer than 2 scores
    return df


def add_history_flags(df, bureau_ids, prev_ids, inst_ids=None):
    """Flag whether each applicant appears in bureau / previous_application (/ installments_payments)."""
    df = df.copy()
    df["HAS_BUREAU_HISTORY"] = df[ID].isin(bureau_ids).astype("int8")
    df["HAS_PREV_APPLICATION"] = df[ID].isin(prev_ids).astype("int8")
    if inst_ids is not None:
        df["HAS_INSTALLMENT_HISTORY"] = df[ID].isin(inst_ids).astype("int8")
    return df


def bureau_features(bureau):
    """Credit Bureau aggregates per applicant (prefix BUR_)."""
    b = pd.DataFrame({
        ID: bureau[ID],
        "active": (bureau["CREDIT_ACTIVE"] == "Active").astype("int8"),
        "credit_sum": bureau["AMT_CREDIT_SUM"].astype("float64"),
        "debt": bureau["AMT_CREDIT_SUM_DEBT"].astype("float64"),
        "overdue": bureau["AMT_CREDIT_SUM_OVERDUE"].astype("float64"),
        "day_overdue": bureau["CREDIT_DAY_OVERDUE"],
        "days_credit": bureau["DAYS_CREDIT"].astype("float64"),
    })
    g = b.groupby(ID)
    out = pd.DataFrame({
        "BUR_LOAN_COUNT": g.size(),
        "BUR_ACTIVE_COUNT": g["active"].sum(),
        "BUR_CREDIT_SUM_SUM": g["credit_sum"].sum(min_count=1),
        "BUR_CREDIT_SUM_MEAN": g["credit_sum"].mean(),
        "BUR_DEBT_SUM": g["debt"].sum(min_count=1),
        "BUR_OVERDUE_SUM": g["overdue"].sum(min_count=1),
        "BUR_MAX_DAY_OVERDUE": g["day_overdue"].max(),
        "BUR_DAYS_CREDIT_MEAN": g["days_credit"].mean(),
    })
    out["BUR_ACTIVE_SHARE"] = safe_div(out["BUR_ACTIVE_COUNT"], out["BUR_LOAN_COUNT"])
    out["BUR_DEBT_CREDIT_RATIO"] = safe_div(out["BUR_DEBT_SUM"], out["BUR_CREDIT_SUM_SUM"])
    return _to_compact(out).reset_index()


def previous_features(prev):
    """Previous Home Credit application aggregates per applicant (prefix PREV_)."""
    status = prev["NAME_CONTRACT_STATUS"]
    p = pd.DataFrame({
        ID: prev[ID],
        "approved": (status == "Approved").astype("int8"),
        "refused": (status == "Refused").astype("int8"),
        "amt_application": prev["AMT_APPLICATION"].astype("float64"),
        "amt_credit": prev["AMT_CREDIT"].astype("float64"),
        # AMT_CREDIT is 0 for many refused/cancelled applications -> ratio NaN there
        "app_credit_ratio": safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"]).astype("float64"),
        "days_decision": prev["DAYS_DECISION"],
    })
    g = p.groupby(ID)
    out = pd.DataFrame({
        "PREV_APP_COUNT": g.size(),
        "PREV_APPROVED_COUNT": g["approved"].sum(),
        "PREV_REFUSED_COUNT": g["refused"].sum(),
        "PREV_AMT_APPLICATION_MEAN": g["amt_application"].mean(),
        "PREV_AMT_CREDIT_MEAN": g["amt_credit"].mean(),
        "PREV_APP_CREDIT_RATIO_MEAN": g["app_credit_ratio"].mean(),
        "PREV_DAYS_SINCE_LAST_DECISION": -g["days_decision"].max(),
    })
    out["PREV_APPROVED_SHARE"] = safe_div(out["PREV_APPROVED_COUNT"], out["PREV_APP_COUNT"])
    out["PREV_REFUSED_SHARE"] = safe_div(out["PREV_REFUSED_COUNT"], out["PREV_APP_COUNT"])
    return _to_compact(out).reset_index()


def installment_features(inst):
    """Repayment behaviour aggregates per applicant (prefix INST_).

    Partial payments of the same instalment are first combined (one row per instalment: amounts
    summed, last payment date kept), so an instalment paid in two parts counts once.
    - days late    = last DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, clipped at 0
    - payment ratio = total AMT_PAYMENT / AMT_INSTALMENT
    - underpaid     = payment ratio < 0.999 (tolerance for float32 rounding)
    - partial payment = instalment settled in more than one payment (extra feature: almost all row-level
      "underpayments" are partial payments that add up to the full instalment)
    """
    keys = [ID, "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
    per = (
        inst.assign(AMT_PAYMENT=inst["AMT_PAYMENT"].astype("float64"),
                    AMT_INSTALMENT=inst["AMT_INSTALMENT"].astype("float64"))
        .groupby(keys, sort=False)
        .agg(amt_instalment=("AMT_INSTALMENT", "max"),
             amt_payment=("AMT_PAYMENT", "sum"),
             n_payments=("AMT_PAYMENT", "count"),
             days_instalment=("DAYS_INSTALMENT", "max"),
             days_entry=("DAYS_ENTRY_PAYMENT", "max"))
        .reset_index()
    )
    # Instalments with no recorded payment row keep a missing payment instead of 0.
    per["amt_payment"] = per["amt_payment"].where(per["n_payments"] > 0)

    days_late = (per["days_entry"].astype("float64") - per["days_instalment"]).clip(lower=0)
    pay_ratio = safe_div(per["amt_payment"], per["amt_instalment"]).astype("float64")
    per = pd.DataFrame({
        ID: per[ID],
        "days_late": days_late,
        "is_late": (days_late > 0).astype("float64").where(days_late.notna()),
        "pay_ratio": pay_ratio,
        "underpaid": (pay_ratio < 0.999).astype("float64").where(pay_ratio.notna()),
        "partial": (per["n_payments"] > 1).astype("float64"),
    })
    g = per.groupby(ID)
    out = pd.DataFrame({
        "INST_COUNT": g.size(),
        "INST_LATE_SHARE": g["is_late"].mean(),
        "INST_DAYS_LATE_MEAN": g["days_late"].mean(),
        "INST_DAYS_LATE_MAX": g["days_late"].max(),
        "INST_PAYMENT_RATIO_MEAN": g["pay_ratio"].mean(),
        "INST_PAYMENT_RATIO_MIN": g["pay_ratio"].min(),
        "INST_UNDERPAID_SHARE": g["underpaid"].mean(),
        "INST_PARTIAL_PAYMENT_SHARE": g["partial"].mean(),
    })
    return _to_compact(out).reset_index()


def build_model_table(app_clean, bureau_agg, prev_agg, inst_agg):
    """Application ratios + left-joined side-table aggregates + history flags (one row per SK_ID_CURR)."""
    df = add_application_ratios(app_clean)
    df = add_history_flags(df, bureau_agg[ID], prev_agg[ID], inst_agg[ID])
    for agg in (bureau_agg, prev_agg, inst_agg):
        df = df.merge(agg, on=ID, how="left", validate="one_to_one")
    for col in COUNT_COLUMNS:
        df[col] = df[col].fillna(0).astype("int32")
    # Left joins turn integer aggregates into float64 where an applicant has no match.
    float64_cols = df.select_dtypes("float64").columns
    df[float64_cols] = df[float64_cols].astype("float32")
    return df


def load_side_tables(columns=None):
    """Load bureau, previous_application and installments_payments Parquet files."""
    columns = columns or {}
    return tuple(
        pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns.get(name))
        for name in ("bureau", "previous_application", "installments_payments")
    )


def _to_compact(df):
    """Downcast aggregate columns: floats -> float32, integer counts -> int32."""
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int32")
    return df
