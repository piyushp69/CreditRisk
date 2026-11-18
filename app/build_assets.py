"""Build everything the website needs into app/assets/ (small CSV / JSON / Parquet) and app/models/.

Run once from the project root after notebooks 01-05 (about 5 minutes):

    python app/build_assets.py

No raw data is shipped: only aggregates, model outputs without SK_ID_CURR, and the final LightGBM as a text file.
Results that notebook 04 printed but did not save (thin-file AUCs, ablation, SHAP, fairness) are recomputed with the
same code, seeds and order, and checked against the notebook values.
"""
import json
import shutil
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path[:0] = [str(ROOT), str(APP_DIR)]

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src.cleaning import clean_application, days_employed_anomaly_report, income_outlier_report
from src.config import FIGURES_DIR, ID, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from src.cost import (BASE_LGD, BASE_MARGIN, approve_all_cost, decision_summary, optimal_threshold,
                      optimal_threshold_table, threshold_sweep)
from src.eda import binned_rate_table, default_rate_table, missing_table
from src.explain import reason_codes
from src.modeling import (METRICS, categorical_columns, cross_validate_gbm, feature_columns, fit_lgbm,
                          load_or_create_split, metric_row, train_holdout)
from utils.predictor import build_features, form_values, score

ASSETS = APP_DIR / "assets"
APP_MODELS = APP_DIR / "models"
MODEL_ORDER = ["Logistic regression", "Decision tree", "Random forest", "XGBoost", "LightGBM"]
SHORT = {"Logistic regression": "logreg", "Decision tree": "dt", "Random forest": "rf", "XGBoost": "xgb",
         "LightGBM": "lgbm"}
EDA_CATEGORICAL = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
                   "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
                   "ORGANIZATION_TYPE", "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "REGION_RATING_CLIENT",
                   "CNT_CHILDREN"]
EDA_NUMERIC = ["AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
               "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO", "EXT_SOURCE_MEAN", "INST_LATE_SHARE",
               "PREV_REFUSED_SHARE", "BUR_ACTIVE_SHARE", "DAYS_LAST_PHONE_CHANGE"]
RATIO_COLS = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO", "EMPLOYED_TO_AGE",
              "INCOME_PER_FAMILY_MEMBER", "EXT_SOURCE_MEAN", "EXT_SOURCE_STD"]
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN", "EXT_SOURCE_STD"]
FIGURES_TO_COPY = ["26_shap_beeswarm.png", "33_decision_tree_top3_levels.png"]

# Values printed by notebook 04 that the recomputation here must reproduce.
NB04 = {"lgbm_auc_no_bureau": 0.7645, "lgbm_ci_no_bureau": (0.7512, 0.7794), "ablation_noext": 0.7548,
        "ablation_nobureau": 0.7783, "ablation_altonly": 0.7373, "altonly_no_bureau": 0.7507,
        "neither_auc": 0.7893, "shap_top": ("EXT_SOURCE_MEAN", 0.3610), "auc_nogender": 0.7803, "t_nogender": 0.72,
        "proxy_auc": 0.9094}


def log(msg, t0=None):
    print(msg + (f"  ({time.time() - t0:.0f}s)" if t0 is not None else ""), flush=True)


def save_csv(df, name, index=False):
    df.to_csv(ASSETS / name, index=index)


def interval_labels(index, decimals=2):
    return [f"{iv.left:,.{decimals}f} – {iv.right:,.{decimals}f}" for iv in index]


def rate_table(df, variable, kind):
    out = df.reset_index().rename(columns={df.index.name or "index": "group"})
    out.insert(0, "variable", variable)
    out.insert(1, "kind", kind)
    out["group"] = out["group"].astype(str)
    return out


def downsample_curve(x, y, n=500):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return x[idx], y[idx]


def main():
    t_start = time.time()
    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "figures").mkdir(exist_ok=True)
    APP_MODELS.mkdir(exist_ok=True)
    facts = {}

    # ------------------------------------------------------------------ data and split
    app_raw = pd.read_parquet(PROCESSED_DIR / "application_train.parquet")
    clean = clean_application(app_raw)
    mt = pd.read_parquet(PROCESSED_DIR / "model_table.parquet")
    split, created = load_or_create_split(mt)
    assert not created, "split must come from notebook 03"
    train, holdout = train_holdout(mt, split)
    features = feature_columns(mt)
    cats = categorical_columns(mt[features])
    log(f"loaded: raw {app_raw.shape}, model table {mt.shape}", t_start)

    # ------------------------------------------------------------------ dataset facts
    target_counts = app_raw[TARGET].value_counts().sort_index()
    miss = missing_table(app_raw)
    building = [c for c in miss.index if c.endswith(("_AVG", "_MODE", "_MEDI"))]
    anom = days_employed_anomaly_report(app_raw)
    anomalous = app_raw["DAYS_EMPLOYED"] == 365243
    inc_threshold, inc_n_above, inc_rate_above, _ = income_outlier_report(clean)
    files = []
    for name, used, role in [
            ("application_train", True, "One row per loan application with the TARGET label (the modelling table)"),
            ("bureau", True, "Previous credits at other institutions reported to the Credit Bureau"),
            ("previous_application", True, "The applicant's previous applications to Home Credit"),
            ("installments_payments", True, "Repayment history of previous Home Credit loans"),
            ("HomeCredit_columns_description", True, "Column dictionary of all tables"),
            ("application_test", False, "Same as train but without TARGET: downloaded, not used for evaluation")]:
        meta = pq.read_metadata(PROCESSED_DIR / f"{name}.parquet")
        files.append({"file": f"{name}.csv", "status": "used" if used else "downloaded, not used",
                      "rows": meta.num_rows, "columns": meta.num_columns, "contents": role})
    for name, reason in [
            ("bureau_balance.csv", "Month-by-month status of each bureau credit; bureau.csv already summarises it"),
            ("POS_CASH_balance.csv", "Monthly snapshots of previous point-of-sale and cash loans; left out of scope"),
            ("credit_card_balance.csv", "Monthly snapshots of previous credit cards; left out of scope"),
            ("sample_submission.csv", "Kaggle submission format only")]:
        files.append({"file": name, "status": "excluded", "rows": None, "columns": None, "contents": reason})
    save_csv(pd.DataFrame(files), "files.csv")

    facts["dataset"] = {
        "n_raw": int(len(app_raw)), "n_raw_columns": int(app_raw.shape[1]), "n_clean": int(len(clean)),
        "n_repaid": int(target_counts[0]), "n_defaults": int(target_counts[1]),
        "default_rate": float(app_raw[TARGET].mean()), "repaid_per_default": float(target_counts[0] / target_counts[1]),
        "xna_rows": int((app_raw["CODE_GENDER"] == "XNA").sum()),
        "n_features": len(features), "n_categorical": len(cats), "n_numeric": len(features) - len(cats),
        "n_train": int(len(train)), "n_holdout": int(len(holdout)),
        "train_default_rate": float(train[TARGET].mean()), "holdout_default_rate": float(holdout[TARGET].mean()),
        "n_holdout_defaults": int(holdout[TARGET].sum()),
        "missing_columns": int(len(miss)), "missing_over_50": int((miss["pct_missing"] > 50).sum()),
        "building_over_50": int(sum(miss.loc[c, "pct_missing"] > 50 for c in building)),
        "max_pct_missing": float(miss["pct_missing"].max()),
        "anomaly_count": int(anomalous.sum()), "anomaly_share": float(anomalous.mean()),
        "anomaly_rate": float(anom["default_rate"].iloc[0]), "valid_rate": float(anom["default_rate"].iloc[1]),
        "anomaly_pensioners": int((app_raw.loc[anomalous, "NAME_INCOME_TYPE"] == "Pensioner").sum()),
        "anomaly_unemployed": int((app_raw.loc[anomalous, "NAME_INCOME_TYPE"] == "Unemployed").sum()),
        "income_median": float(clean["AMT_INCOME_TOTAL"].median()), "income_p999": inc_threshold,
        "income_max": float(clean["AMT_INCOME_TOTAL"].max()), "income_n_above_p999": inc_n_above,
        "income_rate_above_p999": inc_rate_above,
        "n_test_unlabelled": int(pq.read_metadata(PROCESSED_DIR / "application_test.parquet").num_rows),
        "kaggle_url": "https://www.kaggle.com/competitions/home-credit-default-risk",
    }
    facts["config"] = {"random_state": RANDOM_STATE, "holdout_share": len(holdout) / len(mt), "cv_folds": 5,
                       "lgd": BASE_LGD, "margin": BASE_MARGIN, "early_stopping_rounds": 100, "es_fraction": 0.10}
    save_csv(miss.reset_index(names="column").assign(is_building=lambda d: d["column"].isin(building)),
             "missing_values.csv")
    save_csv(anom.reset_index(names="group"), "days_employed_anomaly.csv")

    # column dictionary
    desc = pd.read_parquet(PROCESSED_DIR / "HomeCredit_columns_description.parquet")
    desc = desc.rename(columns={"Row": "column"})[["Table", "column", "Description", "Special"]]
    desc["Table"] = desc["Table"].astype(str).str.replace(".csv", "", regex=False)
    save_csv(desc, "column_dictionary.csv")
    log("dataset facts, missing values, dictionary", t_start)

    # ------------------------------------------------------------------ EDA aggregates
    rows = []
    age_bins = list(range(20, 75, 5))
    age = binned_rate_table(clean, "AGE_YEARS", bins=age_bins, labels=[f"{a}-{a + 4}" for a in age_bins[:-1]])
    rows.append(rate_table(age.rename_axis("group"), "AGE_GROUP", "ordered"))
    for col in EDA_CATEGORICAL:
        ordered = col in ("REGION_RATING_CLIENT", "CNT_CHILDREN")
        src = clean.assign(CNT_CHILDREN=clean["CNT_CHILDREN"].clip(upper=4).map(lambda v: "4+" if v >= 4 else str(v))) \
            if col == "CNT_CHILDREN" else clean
        t = default_rate_table(src, col, sort_by_rate=not ordered)
        rows.append(rate_table(t.rename_axis("group"), col, "ordered" if ordered else "categorical"))
    flags = mt[["HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION", TARGET]].copy()
    for col in ["HAS_BUREAU_HISTORY", "HAS_PREV_APPLICATION"]:
        t = default_rate_table(flags.assign(**{col: flags[col].map({1: "Yes", 0: "No"})}), col)
        rows.append(rate_table(t.rename_axis("group"), col, "history"))
    combo = flags["HAS_BUREAU_HISTORY"].map({1: "bureau", 0: "no bureau"}) + " / " + \
        flags["HAS_PREV_APPLICATION"].map({1: "prev app", 0: "no prev app"})
    t = default_rate_table(flags.assign(HISTORY_GROUP=combo), "HISTORY_GROUP", sort_by_rate=True)
    rows.append(rate_table(t.rename_axis("group"), "HISTORY_GROUP", "history"))
    late = pd.cut(mt["INST_LATE_SHARE"], [-0.001, 0, 0.05, 0.10, 0.20, 0.50, 1.0],
                  labels=["0 (never late)", "0–5%", "5–10%", "10–20%", "20–50%", ">50%"])
    t = default_rate_table(mt.assign(LATE_BAND=late), "LATE_BAND", missing_label="No instalment data")
    rows.append(rate_table(t.rename_axis("group"), "INST_LATE_SHARE_BAND", "ordered"))
    ref = pd.cut(mt["PREV_REFUSED_SHARE"], [-0.001, 0, 0.25, 0.50, 1.0],
                 labels=["0 (none refused)", "0–25%", "25–50%", ">50%"])
    t = default_rate_table(mt.assign(REFUSED_BAND=ref), "REFUSED_BAND", missing_label="No previous application")
    rows.append(rate_table(t.rename_axis("group"), "PREV_REFUSED_SHARE_BAND", "ordered"))
    ratios = mt  # model table already holds the ratios built by src.features.add_application_ratios
    for col in ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EXT_SOURCE_MEAN"]:
        t = binned_rate_table(ratios, col, q=10)
        t.index = [f"D{i} ({lab})" for i, lab in enumerate(interval_labels(t.index, 3 if "EXT" in col else 2), 1)]
        rows.append(rate_table(t.rename_axis("group"), f"{col}_DECILE", "decile"))
    eda_rates = pd.concat(rows, ignore_index=True)
    eda_rates["overall_rate"] = float(clean[TARGET].mean())
    save_csv(eda_rates, "eda_default_rates.csv")

    # binned distributions by TARGET (density per class) and default rate per bin
    hist_rows = []
    for col in EDA_NUMERIC + ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        values = mt[col].astype("float64")
        if col.startswith("EXT_SOURCE"):
            edges = np.linspace(0, 1, 51)
        else:
            lo, hi = values.quantile([0.005, 0.995])
            edges = np.linspace(lo, hi, 41)
        width = np.diff(edges)
        for t in (0, 1):
            v = values[(mt[TARGET] == t) & values.notna()]
            counts, _ = np.histogram(v.clip(edges[0], edges[-1]), bins=edges)
            for i, c in enumerate(counts):
                hist_rows.append({"variable": col, "TARGET": t, "bin_left": edges[i], "bin_right": edges[i + 1],
                                  "density": c / (len(v) * width[i]), "count": int(c)})
    save_csv(pd.DataFrame(hist_rows), "eda_histograms.csv")

    # correlations: notebook 01 (cleaned application columns) and notebook 02 (engineered features)
    num_cols = [c for c in clean.select_dtypes("number").columns if c not in (ID, TARGET, "AGE_YEARS", "YEARS_EMPLOYED")]
    corr = clean[num_cols].astype("float64").corrwith(clean[TARGET].astype("float64"))
    top15 = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
    engineered = RATIO_COLS + [c for c in mt.columns if c.startswith(("BUR_", "PREV_", "INST_", "HAS_"))]
    eng = mt[engineered].astype("float64").corrwith(mt[TARGET].astype("float64"))
    eng15 = eng.reindex(eng.abs().sort_values(ascending=False).index).head(15)
    save_csv(pd.concat([pd.DataFrame({"set": "application columns", "feature": top15.index, "corr": top15.values}),
                        pd.DataFrame({"set": "engineered features", "feature": eng15.index, "corr": eng15.values})]),
             "eda_correlations.csv")
    ext_summary = pd.DataFrame({"pct_missing": clean[EXT_COLS[:3]].isna().mean() * 100,
                                "mean_repaid": clean.loc[clean[TARGET] == 0, EXT_COLS[:3]].mean(),
                                "mean_default": clean.loc[clean[TARGET] == 1, EXT_COLS[:3]].mean()})
    save_csv(ext_summary.reset_index(names="feature"), "eda_ext_sources.csv")
    facts["eda"] = {"overall_rate_clean": float(clean[TARGET].mean()),
                    "no_bureau": int((mt["HAS_BUREAU_HISTORY"] == 0).sum()),
                    "no_prev": int((mt["HAS_PREV_APPLICATION"] == 0).sum()),
                    "neither": int(((mt["HAS_BUREAU_HISTORY"] == 0) & (mt["HAS_PREV_APPLICATION"] == 0)).sum()),
                    "no_bureau_with_prev_share": float(mt.loc[mt["HAS_BUREAU_HISTORY"] == 0, "HAS_PREV_APPLICATION"].mean()),
                    "no_bureau_with_inst_share": float(mt.loc[mt["HAS_BUREAU_HISTORY"] == 0, "HAS_INSTALLMENT_HISTORY"].mean()),
                    "occupation_missing": int(clean["OCCUPATION_TYPE"].isna().sum())}
    log("EDA aggregates", t_start)

    # ------------------------------------------------------------------ model results (notebook 03)
    preds = pd.read_parquet(PROCESSED_DIR / "holdout_predictions.parquet")
    oof = pd.read_parquet(PROCESSED_DIR / "oof_predictions.parquet")
    assert (preds[ID].to_numpy() == holdout[ID].to_numpy()).all() and (oof[ID].to_numpy() == train[ID].to_numpy()).all()
    thresholds = json.loads((MODELS_DIR / "cost_thresholds.json").read_text())["thresholds"]
    cv = pd.read_csv(REPORTS_DIR / "cv_results.csv", index_col="model")
    csv_metrics = pd.read_csv(REPORTS_DIR / "metrics_table_final.csv")
    y_ho, credit_ho = preds[TARGET].to_numpy(), preds["AMT_CREDIT"].to_numpy()
    base_cost = approve_all_cost(y_ho, credit_ho)
    rows = []
    for name in MODEL_ORDER:
        p = preds[f"p_{SHORT[name]}"].to_numpy()
        for t_type, t in [("0.5", 0.5), ("cost-optimal", thresholds[name])]:
            s = decision_summary(y_ho, p, credit_ho, t)
            rows.append({"model": name, "threshold_type": t_type, "threshold": t, **metric_row(y_ho, p, t),
                         "approval_rate": s["approval_rate"], "default_rate_approved": s["default_rate_approved"],
                         "total_cost": s["total_cost"], "saving_vs_approve_all": 1 - s["total_cost"] / base_cost,
                         "CV_AUC_mean": cv.loc[name, "CV_AUC_mean"], "CV_AUC_std": cv.loc[name, "CV_AUC_std"],
                         "train_seconds": cv.loc[name, "train_seconds"], "mean_score": float(p.mean())})
    metrics = pd.DataFrame(rows)
    num = METRICS + ["TN", "FP", "FN", "TP", "approval_rate", "default_rate_approved", "total_cost"]
    diff = np.abs(metrics[num].to_numpy(float) - csv_metrics[num].to_numpy(float)) / np.maximum(1, np.abs(csv_metrics[num].to_numpy(float)))
    assert diff.max() < 1e-5, f"recomputed metrics differ from reports/metrics_table_final.csv ({diff.max()})"
    save_csv(metrics, "metrics.csv")
    save_csv(cv.reset_index(), "cv_results.csv")
    save_csv(pd.read_csv(REPORTS_DIR / "overfitting_check.csv"), "overfitting.csv")
    save_csv(pd.read_csv(REPORTS_DIR / "dt_search_results.csv"), "dt_search.csv")
    save_csv(pd.read_csv(REPORTS_DIR / "rf_search_results.csv"), "rf_search.csv")
    save_csv(pd.read_csv(REPORTS_DIR / "lgbm_search_results.csv"), "lgbm_search.csv")
    shutil.copy(PROCESSED_DIR / "threshold_curves.parquet", ASSETS / "threshold_curves.parquet")
    hold_out = preds.drop(columns=[ID])
    hold_out.to_parquet(ASSETS / "holdout_predictions.parquet", index=False)

    curve_rows = []
    for name in MODEL_ORDER:
        p = preds[f"p_{SHORT[name]}"].to_numpy()
        fpr, tpr, _ = roc_curve(y_ho, p)
        fx, fy = downsample_curve(fpr, tpr)
        curve_rows += [{"model": name, "curve": "ROC", "x": a, "y": b} for a, b in zip(fx, fy)]
        prec, rec, _ = precision_recall_curve(y_ho, p)
        rx, ry = downsample_curve(rec[::-1], prec[::-1])
        curve_rows += [{"model": name, "curve": "PR", "x": a, "y": b} for a, b in zip(rx, ry)]
    save_csv(pd.DataFrame(curve_rows), "roc_pr_points.csv")

    # cost-optimal threshold for every (LGD, margin) on the training OOF predictions (business page)
    lgds, margins = np.round(np.arange(0.20, 0.901, 0.05), 2), np.round(np.arange(0.05, 0.301, 0.01), 2)
    grid = []
    for name in MODEL_ORDER:
        g = optimal_threshold_table(oof[TARGET], oof[f"oof_{SHORT[name]}"], oof["AMT_CREDIT"], lgds, margins)
        base = g[(g["LGD"] == BASE_LGD) & (g["MARGIN"] == BASE_MARGIN)]["optimal_threshold"].iloc[0]
        assert abs(base - thresholds[name]) < 1e-9, (name, base, thresholds[name])
        grid.append(g.assign(model=name))
    save_csv(pd.concat(grid), "threshold_grid.csv")

    params = {name: json.loads((MODELS_DIR / f).read_text()) for name, f in
              [("Decision tree", "dt_best_params.json"), ("Random forest", "rf_best_params.json"),
               ("XGBoost", "xgb_params.json"), ("LightGBM", "lgbm_best_params.json")]}
    params["LightGBM"].pop("features")
    opt = metrics[metrics["threshold_type"] == "cost-optimal"].set_index("model")
    facts["models"] = {"order": MODEL_ORDER, "params": params, "thresholds": thresholds,
                       "best_by_auc": opt["AUC"].idxmax(), "approve_all_cost": base_cost,
                       "approve_all_accuracy": float(1 - y_ho.mean()),
                       "n_inputs": {"Logistic regression": int(joblib.load(MODELS_DIR / "logreg_pipeline.joblib")[-1].n_features_in_),
                                    "Decision tree": int(joblib.load(MODELS_DIR / "dt_pipeline.joblib")[-1].n_features_in_)}}
    facts["models"]["n_inputs"]["Random forest"] = facts["models"]["n_inputs"]["Decision tree"]  # same preprocessing
    log("model metrics, curves, threshold grid", t_start)

    # ------------------------------------------------------------------ clustering (notebook 05)
    for name in ["cluster_profiles", "cluster_centres", "kmeans_k_selection", "cluster_feature_experiment"]:
        save_csv(pd.read_parquet(PROCESSED_DIR / f"{name}.parquet"), f"{name}.csv")
    shutil.copy(PROCESSED_DIR / "cluster_plot_sample.parquet", ASSETS / "cluster_plot_sample.parquet")
    facts["clustering"] = json.loads((PROCESSED_DIR / "clustering_summary.json").read_text())

    # ------------------------------------------------------------------ thin-file analysis (notebook 04, B)
    lgbm = joblib.load(MODELS_DIR / "lgbm_model.joblib")
    cfg = json.loads((MODELS_DIR / "lgbm_best_params.json").read_text())
    best_params = cfg["params"]
    p_lgbm = preds["p_lgbm"].to_numpy()
    assert np.allclose(lgbm.predict_proba(holdout[features])[:, 1], p_lgbm)
    rng = np.random.default_rng(RANDOM_STATE)

    def auc_with_ci(y, p, n_boot=200):  # notebook 04's percentile bootstrap, same global generator and order
        y, p = np.asarray(y), np.asarray(p)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() != y[idx].max():
                boots.append(roc_auc_score(y[idx], p[idx]))
        return roc_auc_score(y, p), np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    t0 = time.time()
    groups = {"All holdout": np.ones(len(holdout), dtype=bool),
              "Bureau history: yes": holdout["HAS_BUREAU_HISTORY"].to_numpy() == 1,
              "Bureau history: no": holdout["HAS_BUREAU_HISTORY"].to_numpy() == 0,
              "Previous application: yes": holdout["HAS_PREV_APPLICATION"].to_numpy() == 1,
              "Previous application: no": holdout["HAS_PREV_APPLICATION"].to_numpy() == 0}
    nb04_models = {"LightGBM": p_lgbm, "Logistic regression": preds["p_logreg"].to_numpy(),
                   "XGBoost": preds["p_xgb"].to_numpy()}
    rows = []
    for gname, mask in groups.items():
        row = {"group": gname, "applicants": int(mask.sum()), "default_rate": float(y_ho[mask].mean())}
        for mname, prob in nb04_models.items():
            auc, lo, hi = auc_with_ci(y_ho[mask], prob[mask])
            row.update({f"{mname} AUC": auc, f"{mname} CI low": lo, f"{mname} CI high": hi})
        rows.append(row)
    group_auc = pd.DataFrame(rows)
    for mname in ["Decision tree", "Random forest"]:  # point estimates only (no extra draws from the generator)
        group_auc[f"{mname} AUC"] = [roc_auc_score(y_ho[m], preds[f"p_{SHORT[mname]}"].to_numpy()[m])
                                     for m in groups.values()]
    nb_row = group_auc.set_index("group").loc["Bureau history: no"]
    assert round(nb_row["LightGBM AUC"], 4) == NB04["lgbm_auc_no_bureau"]
    assert (round(nb_row["LightGBM CI low"], 4), round(nb_row["LightGBM CI high"], 4)) == NB04["lgbm_ci_no_bureau"]
    save_csv(group_auc, "thin_file_auc.csv")

    bur_cols = [c for c in features if c.startswith("BUR_")] + ["HAS_BUREAU_HISTORY"]
    variants = {"Full model": [], "No EXT_SOURCE": EXT_COLS, "No bureau": bur_cols,
                "Alternative data only": EXT_COLS + bur_cols}
    no_bureau, has_bureau = groups["Bureau history: no"], groups["Bureau history: yes"]
    rows = []
    for name, drop in variants.items():
        feats = [f for f in features if f not in drop]
        model = lgbm if not drop else fit_lgbm(best_params, train[feats], train[TARGET])
        prob = model.predict_proba(holdout[feats])[:, 1]
        a, lo, hi = auc_with_ci(y_ho, prob)
        anb, lonb, hinb = auc_with_ci(y_ho[no_bureau], prob[no_bureau])
        rows.append({"variant": name, "n_features": len(feats), "AUC all": a, "all CI low": lo, "all CI high": hi,
                     "AUC no-bureau": anb, "no-bureau CI low": lonb, "no-bureau CI high": hinb,
                     "AUC with-bureau": roc_auc_score(y_ho[has_bureau], prob[has_bureau])})
    ablation = pd.DataFrame(rows)
    ab = ablation.set_index("variant")
    assert round(ab.loc["No EXT_SOURCE", "AUC all"], 4) == NB04["ablation_noext"]
    assert round(ab.loc["No bureau", "AUC all"], 4) == NB04["ablation_nobureau"]
    assert round(ab.loc["Alternative data only", "AUC all"], 4) == NB04["ablation_altonly"]
    assert round(ab.loc["Alternative data only", "AUC no-bureau"], 4) == NB04["altonly_no_bureau"]
    save_csv(ablation, "ablation.csv")

    neither = no_bureau & groups["Previous application: no"]
    rng = np.random.default_rng(RANDOM_STATE)  # notebook 04 uses a fresh generator here
    auc_n, lo_n, hi_n = auc_with_ci(y_ho[neither], p_lgbm[neither])
    assert round(auc_n, 4) == NB04["neither_auc"]
    facts["thin_file"] = {"neither_holdout": int(neither.sum()), "neither_holdout_defaults": int(y_ho[neither].sum()),
                          "neither_auc": auc_n, "neither_ci": [lo_n, hi_n], "n_boot": 200}
    log("thin-file AUCs and ablation reproduce notebook 04", t0)

    # ------------------------------------------------------------------ SHAP (notebook 04, C) via pred_contrib
    t0 = time.time()
    shap_rows = holdout.sample(5000, random_state=RANDOM_STATE)
    X_shap = shap_rows[features]
    contrib = lgbm.predict(X_shap, pred_contrib=True)
    values = pd.DataFrame(contrib[:, :-1], columns=features, index=X_shap.index)
    mean_abs = values.abs().mean().sort_values(ascending=False)
    assert mean_abs.index[0] == NB04["shap_top"][0] and round(mean_abs.iloc[0], 4) == NB04["shap_top"][1]
    save_csv(mean_abs.head(25).rename("mean_abs_shap").rename_axis("feature").reset_index(), "shap_importance.csv")
    top_num = [f for f in mean_abs.head(8).index if f not in cats]
    dec = {f: values[f].groupby(pd.qcut(X_shap[f].rank(method="first"), 10,
                                        labels=[f"D{i}" for i in range(1, 11)]), observed=True).mean()
           for f in top_num}
    direction = pd.DataFrame(dec).T
    direction.insert(0, "spearman", [X_shap[f].corr(values[f], method="spearman") for f in top_num])
    save_csv(direction.reset_index(names="feature"), "shap_direction.csv")
    gender_shap = values["CODE_GENDER"].groupby(X_shap["CODE_GENDER"].astype(str)).mean()
    facts["shap"] = {"sample": len(X_shap), "base_value": float(contrib[0, -1]),
                     "gender_mean_shap": {k: float(v) for k, v in gender_shap.items()}}

    t_star = thresholds["LightGBM"]
    p_shap = lgbm.predict_proba(X_shap)[:, 1]
    rejected = np.where(p_shap >= t_star)[0]
    order = rejected[np.argsort(p_shap[rejected])]
    picks = {"Highest risk": order[-1], "Median rejected": order[len(order) // 2], "Closest to the threshold": order[0]}
    rc, wf = [], []
    for k, (label, i) in enumerate(picks.items(), 1):
        codes = reason_codes(values.iloc[i].to_numpy(), X_shap.iloc[i], top_n=3)
        rc += [{"applicant": k, "label": label, "score": float(p_shap[i]), "actual": int(shap_rows[TARGET].iloc[i]),
                **r} for r in codes.to_dict("records")]
        top = values.iloc[i].reindex(values.iloc[i].abs().sort_values(ascending=False).index)
        wf += [{"applicant": k, "feature": f, "shap": float(v), "value": str(X_shap.iloc[i][f])}
               for f, v in top.head(10).items()]
        wf.append({"applicant": k, "feature": "Other features", "shap": float(top.iloc[10:].sum()), "value": ""})
    save_csv(pd.DataFrame(rc), "reason_codes.csv")
    save_csv(pd.DataFrame(wf), "reason_waterfalls.csv")
    facts["shap"]["rejected_in_sample"] = int(len(rejected))
    log("SHAP importance, directions, reason codes", t0)

    # ------------------------------------------------------------------ fairness (notebook 04, D)
    t0 = time.time()
    age_bins, age_labels = [20, 30, 40, 50, 60, 70], ["20-29", "30-39", "40-49", "50-59", "60-69"]

    def fairness_table(df, col, prob, threshold):
        d = pd.DataFrame({"group": df[col].astype(str).to_numpy(), "y": df[TARGET].to_numpy(), "approve": prob < threshold})
        g = d.groupby("group", observed=True)
        return pd.DataFrame({
            "applicants": g.size(), "actual_default_rate": g["y"].mean(), "approval_rate": g["approve"].mean(),
            "default_rate_approved": d[d["approve"]].groupby("group", observed=True)["y"].mean(),
            "good_applicants_rejected": d[d["y"] == 0].groupby("group", observed=True)["approve"].apply(lambda a: 1 - a.mean()),
        })

    holdout_f = holdout.assign(AGE_GROUP=pd.cut(holdout["AGE_YEARS"], age_bins, right=False, labels=age_labels))
    feats_ng = [f for f in features if f != "CODE_GENDER"]
    cv_ng = cross_validate_gbm(fit_lgbm, best_params, train[feats_ng], train[TARGET])
    t_ng = optimal_threshold(threshold_sweep(train[TARGET].to_numpy(), cv_ng["oof"], train["AMT_CREDIT"].to_numpy()))
    lgbm_ng = fit_lgbm(best_params, train[feats_ng], train[TARGET])
    p_ng = lgbm_ng.predict_proba(holdout[feats_ng])[:, 1]
    assert round(roc_auc_score(y_ho, p_ng), 4) == NB04["auc_nogender"] and t_ng == NB04["t_nogender"]
    fair = []
    for model, prob, t in [("with CODE_GENDER", p_lgbm, t_star), ("without CODE_GENDER", p_ng, t_ng)]:
        for col, label in [("CODE_GENDER", "Gender"), ("AGE_GROUP", "Age group")]:
            ft = fairness_table(holdout_f, col, prob, t).reset_index()
            fair.append(ft.assign(model=model, attribute=label, threshold=t))
    fair = pd.concat(fair, ignore_index=True)
    save_csv(fair, "fairness.csv")
    y_gender_tr = (train["CODE_GENDER"] == "M").astype(int)
    proxy = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
    proxy.fit(train[feats_ng], y_gender_tr)
    proxy_auc = roc_auc_score((holdout["CODE_GENDER"] == "M").astype(int), proxy.predict_proba(holdout[feats_ng])[:, 1])
    assert round(proxy_auc, 4) == NB04["proxy_auc"]
    gain = pd.Series(proxy.booster_.feature_importance("gain"), index=feats_ng)
    save_csv((gain / gain.sum()).sort_values(ascending=False).head(10).rename("gain_share").rename_axis("feature").reset_index(),
             "proxy_features.csv")
    facts["fairness"] = {
        "auc_with": float(roc_auc_score(y_ho, p_lgbm)), "auc_without": float(roc_auc_score(y_ho, p_ng)),
        "cv_auc_with": cfg["cv_auc_mean"], "cv_auc_without": cv_ng["auc_mean"], "t_with": t_star, "t_without": t_ng,
        "cost_with": float(decision_summary(y_ho, p_lgbm, credit_ho, t_star)["total_cost"]),
        "cost_without": float(decision_summary(y_ho, p_ng, credit_ho, t_ng)["total_cost"]),
        "proxy_auc": float(proxy_auc)}
    log("fairness tables and no-gender model reproduce notebook 04", t0)

    # ------------------------------------------------------------------ live predictor: model, defaults, examples
    booster_path = APP_MODELS / "lgbm_model.txt"
    lgbm.booster_.save_model(str(booster_path), num_iteration=lgbm.best_iteration_)
    booster = lgb.Booster(model_file=str(booster_path))
    assert np.allclose(booster.predict(holdout[features]), p_lgbm)
    categories = {c: [str(v) for v in train[c].cat.categories] for c in cats}
    defaults = {}
    for f in features:
        if f in cats:
            defaults[f] = str(train[f].mode().iloc[0])
        else:
            defaults[f] = float(train[f].astype("float64").median())
    defaults["DAYS_EMPLOYED_ANOM"] = 0
    defaults["DAYS_EMPLOYED"] = float(train["DAYS_EMPLOYED"].median())
    ranges = {col: [float(train[col].quantile(0.01)), float(train[col].quantile(0.99))]
              for col in ["AGE_YEARS", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
                          "YEARS_EMPLOYED", "EXT_SOURCE_MEAN", "PREV_REFUSED_COUNT", "INST_LATE_SHARE"]}

    # Example applicants: real holdout profiles with every form field present, closest to the 10th / 70th / 97th
    # percentile of the holdout scores (clearly approved, approved, clearly rejected at the cost-optimal threshold).
    complete = holdout[features].notna()[["AMT_ANNUITY", "AMT_GOODS_PRICE", "EXT_SOURCE_MEAN", "YEARS_EMPLOYED",
                                         "INST_LATE_SHARE", "NAME_EDUCATION_TYPE"]].all(axis=1).to_numpy()
    examples = []
    for label, q in [("Low risk", 0.10), ("Medium risk", 0.70), ("High risk", 0.97)]:
        target_score = np.quantile(p_lgbm, q)
        cand = np.where(complete)[0]
        i = cand[np.argmin(np.abs(p_lgbm[cand] - target_score))]
        row = {f: (str(v) if f in cats else float(v)) for f, v in holdout[features].iloc[i].items()}
        X = build_features(row, form_values(row), features, categories)
        s, _, _ = score(booster, X)
        assert abs(s - p_lgbm[i]) < 1e-9, (label, s, p_lgbm[i])  # the form pipeline reproduces the real score
        examples.append({"label": label, "quantile": q, "score": float(p_lgbm[i]), "actual": int(y_ho[i]), "row": row})
    default_X = build_features(defaults, form_values(defaults), features, categories)
    default_score, _, _ = score(booster, default_X)
    (ASSETS / "predictor.json").write_text(json.dumps(
        {"features": features, "categories": categories, "defaults": defaults, "ranges": ranges,
         "examples": examples, "threshold": t_star, "default_score": default_score}, indent=1))
    log("predictor: booster text model, defaults, 3 examples (scores reproduced exactly)")

    # ------------------------------------------------------------------ feature dictionary, figures, facts
    fe = pd.read_csv(APP_DIR / "feature_dictionary_source.csv")
    missing_doc = [f for f in fe["feature"] if f not in features]
    assert not missing_doc, f"documented features not in the model table: {missing_doc}"
    save_csv(fe, "feature_dictionary.csv")
    for f in FIGURES_TO_COPY:
        shutil.copy(FIGURES_DIR / f, ASSETS / "figures" / f)
    facts["built"] = time.strftime("%Y-%m-%d %H:%M")
    (ASSETS / "facts.json").write_text(json.dumps(facts, indent=1, default=float))

    size = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file()) / 1024 ** 2
    model_size = booster_path.stat().st_size / 1024 ** 2
    log(f"done: app/assets {size:.1f} MB, app/models/lgbm_model.txt {model_size:.1f} MB", t_start)
    assert size < 50 and model_size < 100


if __name__ == "__main__":
    main()
