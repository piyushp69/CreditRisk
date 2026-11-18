"""K-Means segmentation helpers used in notebook 05. Every transformer is fitted on the training set only."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

# Applicant-profile features: demographics, affordability, external score and past behaviour with Home Credit.
CLUSTER_FEATURES = ["AGE_YEARS", "LOG_INCOME", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "YEARS_EMPLOYED",
                    "EXT_SOURCE_MEAN", "INST_LATE_SHARE", "PREV_REFUSED_SHARE"]


def cluster_frame(df):
    """The clustering features; income enters as log(AMT_INCOME_TOTAL) because of its extreme right skew."""
    out = pd.DataFrame(index=df.index)
    for col in CLUSTER_FEATURES:
        out[col] = (np.log(df["AMT_INCOME_TOTAL"].astype("float64")) if col == "LOG_INCOME"
                    else df[col].astype("float64"))
    return out


def make_cluster_scaler():
    """Median imputation + standardisation (K-Means uses Euclidean distance, so every feature needs the same scale)."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])


def make_kmeans(k):
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
