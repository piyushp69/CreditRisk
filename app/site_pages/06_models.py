import pandas as pd
import streamlit as st

from utils import charts as C
from utils import content
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds, cfg, mods = F["dataset"], F["config"], F["models"]
params = mods["params"]
over = L.csv("overfitting.csv").set_index("model")
points = L.csv("roc_pr_points.csv")
best = L.best_model()

ui.header("How each model works", f"{len(L.MODEL_ORDER)} models were trained on the same {ds['n_train']:,} applicants and "
          f"{ds['n_features']} features, tuned with cross-validation on the training set, and evaluated **once** on "
          f"the {ds['n_holdout']:,} holdout applicants. Pick a model below.", kicker="Models")
t_type = st.segmented_control("Threshold for the decision metrics and confusion matrix",
                              ["cost-optimal", "0.5"], default="cost-optimal",
                              format_func=lambda t: "Each model's cost-optimal threshold" if t == "cost-optimal"
                              else "Conventional 0.5")
t_type = t_type or "cost-optimal"


def settings_table(name):
    if name == "Logistic regression":
        rows = [("Solver", "lbfgs, max_iter = 2000"), ("Regularisation", "L2 (sklearn default strength)"),
                ("Class weight", "balanced"), ("Model inputs", mods["n_inputs"][name])]
    elif name == "Decision tree":
        p = params[name]
        rows = [("max_depth", p["params"]["max_depth"]), ("min_samples_leaf", p["params"]["min_samples_leaf"]),
                ("Leaves in the final tree", p["leaves"]), ("Class weight", p["class_weight"]),
                ("Model inputs", mods["n_inputs"][name])]
    elif name == "Random forest":
        p = params[name]
        rows = [(k, "None (grow fully)" if v is None else v) for k, v in p["params"].items()] + \
               [("max_samples", p["max_samples"]), ("Class weight", p["class_weight"]),
                ("Configurations tried", p["trials_run"])]
    elif name == "XGBoost":
        p = params[name]
        rows = list(p["params"].items()) + [("Trees (early stopping)", p["final_best_iteration"]),
                                            ("Imbalance", "scale_pos_weight = repaid / defaults")]
    else:
        p = params[name]
        rows = list(p["params"].items()) + [("Trees (early stopping)", p["final_best_iteration"]),
                                            ("Imbalance", "scale_pos_weight = repaid / defaults")]
    return pd.DataFrame(rows, columns=["Setting", "Value"]).astype(str)


tabs = st.tabs(L.MODEL_ORDER)
for tab, name in zip(tabs, L.MODEL_ORDER):
    info = content.MODELS[name]
    m = L.metrics_at(t_type).loc[name]
    o = over.loc[name]
    with tab:
        st.markdown(f"### {name}  \n*{info['family']}*")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**How it works.** {info['how']}")
            st.markdown(f"**In one picture:** {info['analogy']}")
            ui.diagram(info["dot"])
        with c2:
            st.markdown(f"**Why it was chosen.** {info['why']}")
            st.markdown(f"**Preprocessing needed.** {info['preprocessing']}")
            st.markdown(f"**Class imbalance.** {info['imbalance']}")
            st.markdown(f"**Tuning.** {info['tuning']}")

        st.markdown("#### Final settings")
        c1, c2 = st.columns([2, 3])
        with c1:
            ui.table(settings_table(name))
        with c2:
            search = {"Decision tree": "dt_search.csv", "Random forest": "rf_search.csv",
                      "LightGBM": "lgbm_search.csv"}.get(name)
            if search:
                res = L.csv(search)
                score_col = [c for c in res.columns if c.startswith("cv") and c.endswith("mean")][0]
                st.caption(f"Tuning results on the training set ({len(res)} configurations, best first; "
                           f"`{score_col}` = cross-validated AUC)")
                ui.table(res.sort_values(score_col, ascending=False).head(8).round(4), height=320)
            else:
                st.caption("This model was not tuned: its settings are fixed (see the table).")

        st.markdown("#### Results on the holdout")
        ui.metrics_row([
            ("AUC-ROC", S.f4(m["AUC"]), content.METRIC_HELP["AUC"]),
            ("Gini", S.f4(m["Gini"]), content.METRIC_HELP["Gini"]),
            ("PR-AUC", S.f4(m["PR_AUC"]), content.METRIC_HELP["PR_AUC"]),
            ("KS", S.f4(m["KS"]), content.METRIC_HELP["KS"]),
            ("CV AUC (train)", f"{m['CV_AUC_mean']:.4f} ± {m['CV_AUC_std']:.4f}", "5-fold cross-validation on the training set"),
        ])
        ui.metrics_row([
            ("Threshold", f"{m['threshold']:.2f}", "Reject when score ≥ threshold"),
            ("Accuracy", S.f4(m["Accuracy"]), content.METRIC_HELP["Accuracy"]),
            ("Precision", S.f4(m["Precision"]), content.METRIC_HELP["Precision"]),
            ("Recall", S.f4(m["Recall"]), content.METRIC_HELP["Recall"]),
            ("F1", S.f4(m["F1"]), content.METRIC_HELP["F1"]),
        ])
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(C.confusion(m["TN"], m["FP"], m["FN"], m["TP"]),
                     title=f"Confusion matrix at threshold {m['threshold']:.2f}",
                     subtitle="Colour = share of the actual class", key=f"cm_{name}")
        with c2:
            ui.chart(C.curves(points, "ROC", highlight=name, height=330), title="ROC curve (other models in grey)",
                     key=f"roc_{name}")
        caught = m["TP"] / (m["TP"] + m["FN"])
        ui.takeaway(f"At threshold {m['threshold']:.2f} the {name.lower()} catches {int(m['TP']):,} of "
                    f"{int(m['TP'] + m['FN']):,} holdout defaulters ({caught:.1%}) and wrongly rejects "
                    f"{int(m['FP']):,} good applicants; overall accuracy is {m['Accuracy']:.2%}. Its holdout AUC "
                    f"({m['AUC']:.4f}) is {'the best of the ' + str(len(L.MODEL_ORDER)) if name == best else 'below the best model'}.")

        if name == "Decision tree":
            st.markdown("#### The tree itself (top three levels)")
            st.image(str(L.figure("33_decision_tree_top3_levels.png")), width="stretch",
                     caption="sklearn plot_tree of the final tree; colour = predicted class, value = class-weighted "
                             "share [repaid, default].")
            dts = L.csv("dt_search.csv")
            leaf = params[name]["params"]["min_samples_leaf"]
            d = dts[dts["min_samples_leaf"] == leaf].sort_values("max_depth")
            color = S.model_colors()[name]
            ui.chart(C.lines(d["max_depth"], [("Training folds", d["train_auc_mean"], S.muted()),
                                              ("Validation folds", d["cv3_auc_mean"], color)],
                             "max_depth (complexity →)", "AUC", marker_x=params[name]["params"]["max_depth"],
                             marker_label="chosen", yfmt=".4f"),
                     title=f"Deeper trees fit the training folds better but validate worse (min_samples_leaf = {leaf})",
                     data=d, key="dt_depth")
            dmax = dts.loc[dts["cv3_auc_mean"].idxmin()]
            ui.takeaway(f"Validation AUC peaks at depth {params[name]['params']['max_depth']}; the most complex tree "
                        f"tried (depth {int(dmax['max_depth'])}, leaves of {int(dmax['min_samples_leaf'])}) scores "
                        f"{dmax['train_auc_mean']:.4f} on its training folds but only {dmax['cv3_auc_mean']:.4f} on "
                        "unseen folds: a textbook case of overfitting (high variance).")

        st.markdown("#### Strengths, weaknesses and bias / variance")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in info["strengths"]))
        with c2:
            st.markdown("**Weaknesses**\n" + "\n".join(f"- {s}" for s in info["weaknesses"]))
        with c3:
            st.markdown(f"**Bias / variance**\n- Train AUC {o['train_AUC']:.4f}\n- CV AUC {o['CV_AUC_mean']:.4f}\n"
                        f"- Holdout AUC {o['holdout_AUC']:.4f}\n- Train − CV gap **{o['train_minus_CV']:+.4f}**\n"
                        f"- Training time {m['train_seconds']:.1f} s")
