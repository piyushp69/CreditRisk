"""Train/holdout split, model definitions, cross-validation and evaluation metrics.

Leakage rules enforced here:
- the holdout split is created once (stratified, random_state=42) and saved to data/processed/split_ids.parquet;
- all preprocessing for the logistic regression, decision tree and random forest lives inside an sklearn Pipeline
  (fitted per CV fold);
- gradient-boosting early stopping uses a validation set carved out of the training data only.
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import ID, PROCESSED_DIR, RANDOM_STATE, TARGET

SPLIT_PATH = PROCESSED_DIR / "split_ids.parquet"
EARLY_STOPPING_ROUNDS = 100
ES_FRACTION = 0.10  # share of the training data held back for early stopping

# Fixed LightGBM settings; tuned parameters are passed separately.
# sklearn-API names: colsample_bytree = feature_fraction, subsample = bagging_fraction.
# metric="auc" only: with scale_pos_weight the default logloss on the (unweighted) early-stopping set worsens
# from the first round and would stop training after one tree.
LGBM_FIXED = dict(objective="binary", metric="auc", n_estimators=10_000, subsample_freq=1, random_state=RANDOM_STATE,
                  n_jobs=-1, deterministic=True, force_col_wise=True, verbose=-1)

XGB_FIXED = dict(n_estimators=10_000, tree_method="hist", enable_categorical=True, eval_metric="auc",
                 early_stopping_rounds=EARLY_STOPPING_ROUNDS, random_state=RANDOM_STATE, n_jobs=-1)


# ---------------------------------------------------------------- split

def make_split(df, test_size=0.20):
    """Stratified train/holdout split of SK_ID_CURR."""
    train_ids, holdout_ids = train_test_split(df[ID], test_size=test_size, stratify=df[TARGET],
                                              random_state=RANDOM_STATE)
    split = pd.concat([pd.DataFrame({ID: train_ids, "SET": "train"}),
                       pd.DataFrame({ID: holdout_ids, "SET": "holdout"})])
    return split.sort_values(ID).reset_index(drop=True)


def load_or_create_split(df):
    """Create the split once and save it; afterwards always reload the saved ids."""
    if SPLIT_PATH.exists():
        split = pd.read_parquet(SPLIT_PATH)
        if set(split[ID]) != set(df[ID]):
            raise ValueError("split_ids.parquet does not match the model table; delete it to re-create")
        return split, False
    split = make_split(df)
    split.to_parquet(SPLIT_PATH, index=False)
    return split, True


def train_holdout(df, split):
    """Return (train, holdout) DataFrames according to the saved split."""
    holdout_ids = set(split.loc[split["SET"] == "holdout", ID])
    is_holdout = df[ID].isin(holdout_ids)
    return df[~is_holdout].reset_index(drop=True), df[is_holdout].reset_index(drop=True)


def feature_columns(df, exclude=()):
    return [c for c in df.columns if c not in (ID, TARGET) and c not in set(exclude)]


def categorical_columns(X):
    return X.select_dtypes("category").columns.tolist()


# ---------------------------------------------------------------- logistic regression

def to_object(X):
    """Category / string columns -> object, so SimpleImputer and OneHotEncoder treat NaN uniformly."""
    return X.astype(object)


def categorical_block():
    """Categoricals: missing -> 'MISSING' level, then one-hot (levels under 1% pooled, unseen levels ignored)."""
    return Pipeline([
        ("to_object", FunctionTransformer(to_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])


def make_scaled_preprocessor(num_cols, cat_cols):
    """Preprocessing for the scale-sensitive logistic regression: median imputation + missing-value indicators and
    standardisation for numeric columns, one-hot encoding for categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical_block(), cat_cols)])


def make_logreg_pipeline(num_cols, cat_cols):
    """Scorecard-style logistic regression: median imputation + missing indicators, scaling, one-hot."""
    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
    return Pipeline([("pre", make_scaled_preprocessor(num_cols, cat_cols)), ("model", model)])


# ---------------------------------------------------------------- decision tree and random forest

def make_tree_preprocessor(num_cols, cat_cols):
    """Preprocessing for sklearn trees: median imputation for numeric columns (splits are thresholds, so no scaling)
    and one-hot encoding for categoricals (sklearn trees need numeric input)."""
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), num_cols),
                              ("cat", categorical_block(), cat_cols)])


DT_FIXED = dict(class_weight="balanced", random_state=RANDOM_STATE)
RF_FIXED = dict(class_weight="balanced_subsample", n_jobs=-1, max_samples=0.5, random_state=RANDOM_STATE)


def make_dt_pipeline(num_cols, cat_cols, **params):
    """Single decision tree; `balanced` class weights make each default count ~11x a repaid loan."""
    return Pipeline([("pre", make_tree_preprocessor(num_cols, cat_cols)),
                     ("model", DecisionTreeClassifier(**{**DT_FIXED, **params}))])


def make_rf_pipeline(num_cols, cat_cols, **params):
    """Random forest; `balanced_subsample` re-weights the classes inside every tree's bootstrap sample."""
    return Pipeline([("pre", make_tree_preprocessor(num_cols, cat_cols)),
                     ("model", RandomForestClassifier(**{**RF_FIXED, **params}))])


# ---------------------------------------------------------------- gradient boosting

def _carve_early_stopping_set(X, y):
    return train_test_split(X, y, test_size=ES_FRACTION, stratify=y, random_state=RANDOM_STATE)


def scale_pos_weight(y):
    return float((y == 0).sum() / (y == 1).sum())


def fit_lgbm(params, X, y):
    """LightGBM with native categoricals, scale_pos_weight = negatives/positives, early stopping on a
    stratified 10% validation set carved from (X, y)."""
    X_fit, X_es, y_fit, y_es = _carve_early_stopping_set(X, y)
    model = lgb.LGBMClassifier(**LGBM_FIXED, **params, scale_pos_weight=scale_pos_weight(y_fit))
    model.fit(X_fit, y_fit, eval_X=X_es, eval_y=y_es,
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return model


def fit_xgb(params, X, y):
    """XGBoost (hist, native categoricals) with the same imbalance handling and early-stopping scheme."""
    X_fit, X_es, y_fit, y_es = _carve_early_stopping_set(X, y)
    model = XGBClassifier(**XGB_FIXED, **params, scale_pos_weight=scale_pos_weight(y_fit))
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


def best_iteration(model):
    if isinstance(model, lgb.LGBMClassifier):
        return int(model.best_iteration_)
    return int(model.best_iteration) + 1


def cross_validate_sklearn(estimator, X, y, n_splits=5, n_jobs=1):
    """Stratified k-fold CV of an sklearn estimator/pipeline (cloned and refitted per fold). Same output as
    `cross_validate_gbm`, including the out-of-fold predictions used to choose decision thresholds.
    `n_jobs` runs folds in parallel; keep 1 for estimators that already use all cores."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.time()

    def run(tr_idx, va_idx):
        model = clone(estimator).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        return va_idx, model.predict_proba(X.iloc[va_idx])[:, 1]

    results = Parallel(n_jobs=n_jobs)(delayed(run)(tr, va) for tr, va in skf.split(X, y))
    oof = np.zeros(len(y), dtype="float64")
    aucs = []
    for va_idx, prob in results:
        oof[va_idx] = prob
        aucs.append(roc_auc_score(y.iloc[va_idx], prob))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "fold_aucs": aucs, "oof": oof,
            "seconds": time.time() - t0}


def cross_validate_gbm(fit_fn, params, X, y, n_splits=5):
    """Stratified k-fold CV. Each fold model early-stops on a carve-out of that fold's training part, so the
    out-of-fold predictions are never used for fitting or stopping."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype="float64")
    aucs, iters = [], []
    t0 = time.time()
    for tr_idx, va_idx in skf.split(X, y):
        model = fit_fn(params, X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va_idx], oof[va_idx]))
        iters.append(best_iteration(model))
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "fold_aucs": aucs,
            "best_iterations": iters, "oof": oof, "seconds": time.time() - t0}


# ---------------------------------------------------------------- metrics

def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov statistic: max distance between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_prob, threshold=0.5):
    """Ranking metrics plus classification metrics for the default class at `threshold`."""
    return {**ranking_metrics(y_true, y_prob), **threshold_metrics(y_true, y_prob, threshold),
            **confusion_counts(y_true, y_prob, threshold)}


def ranking_metrics(y_true, y_prob):
    """Threshold-free metrics: how well defaulters are ranked above non-defaulters."""
    auc = roc_auc_score(y_true, y_prob)
    return {"AUC": auc, "Gini": 2 * auc - 1, "PR_AUC": average_precision_score(y_true, y_prob),
            "KS": ks_statistic(y_true, y_prob)}


def threshold_metrics(y_true, y_prob, threshold, label=None):
    """Precision, recall, F1 (default class) and accuracy when rejecting at `prob >= threshold`.
    Keys are suffixed with `@label` (default: the threshold value)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tag = threshold if label is None else label
    return {
        f"Precision@{tag}": precision_score(y_true, y_pred, zero_division=0),
        f"Recall@{tag}": recall_score(y_true, y_pred, zero_division=0),
        f"F1@{tag}": f1_score(y_true, y_pred, zero_division=0),
        f"Accuracy@{tag}": accuracy_score(y_true, y_pred),
    }


def confusion_counts(y_true, y_prob, threshold, label=None):
    """TN, FP, FN, TP at `threshold` (keys suffixed with `@label` if given)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    suffix = "" if label is None else f"@{label}"
    return {f"TN{suffix}": int(tn), f"FP{suffix}": int(fp), f"FN{suffix}": int(fn), f"TP{suffix}": int(tp)}


METRICS = ["Accuracy", "Precision", "Recall", "F1", "LogLoss", "AUC", "Gini", "PR_AUC", "KS"]


def metric_row(y_true, y_prob, threshold):
    """All comparison metrics at one threshold with plain keys (METRICS order) plus TN, FP, FN, TP."""
    tm = threshold_metrics(y_true, y_prob, threshold, label="t")
    return {"Accuracy": tm["Accuracy@t"], "Precision": tm["Precision@t"], "Recall": tm["Recall@t"],
            "F1": tm["F1@t"], "LogLoss": log_loss_metric(y_true, y_prob), **ranking_metrics(y_true, y_prob),
            **confusion_counts(y_true, y_prob, threshold)}


def log_loss_metric(y_true, y_prob):
    """Log loss of the scores read as probabilities (clipped away from 0 and 1). Class-weighted models inflate their
    scores on purpose, so their log loss is high even when they rank applicants well."""
    p = np.clip(np.asarray(y_prob, dtype="float64"), 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, p, labels=[0, 1]))
