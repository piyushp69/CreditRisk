"""Summary tables used in EDA: missing values and default rate by group."""
import pandas as pd

from .config import TARGET


def missing_table(df):
    """Count and % missing per column, sorted descending (columns with no missing values excluded)."""
    n_missing = df.isna().sum()
    out = pd.DataFrame({"n_missing": n_missing, "pct_missing": n_missing / len(df) * 100})
    return out[out["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def default_rate_table(df, col, target=TARGET, missing_label="Missing", sort_by_rate=False):
    """Applicants, share, defaults and default rate per value of `col` (NaN kept as its own group)."""
    key = df[col]
    if isinstance(key.dtype, pd.CategoricalDtype):
        key = key.cat.add_categories([missing_label]) if missing_label not in key.cat.categories else key
        key = key.fillna(missing_label)
    elif key.dtype == object or isinstance(key.dtype, pd.StringDtype):
        key = key.fillna(missing_label)
    grouped = df[target].groupby(key, observed=True, dropna=False)
    out = pd.DataFrame({"applicants": grouped.size(), "defaults": grouped.sum()})
    out["share"] = out["applicants"] / out["applicants"].sum()
    out["default_rate"] = out["defaults"] / out["applicants"]
    out = out[["applicants", "share", "defaults", "default_rate"]]
    if sort_by_rate:
        out = out.sort_values("default_rate", ascending=False)
    return out


def binned_rate_table(df, col, target=TARGET, q=10, bins=None, labels=None, right=False):
    """Default rate by quantile bins (deciles by default) or explicit `bins` of a numeric column.

    Rows where `col` is missing are excluded.
    """
    sub = df[[col, target]].dropna(subset=[col])
    if bins is None:
        binned = pd.qcut(sub[col], q=q, duplicates="drop")
    else:
        binned = pd.cut(sub[col], bins=bins, labels=labels, right=right, include_lowest=True)
    grouped = sub[target].groupby(binned, observed=True)
    out = pd.DataFrame({"applicants": grouped.size(), "defaults": grouped.sum()})
    out["share"] = out["applicants"] / out["applicants"].sum()
    out["default_rate"] = out["defaults"] / out["applicants"]
    return out[["applicants", "share", "defaults", "default_rate"]]
