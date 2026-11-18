import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.thresholds import metrics_by_threshold
from utils import charts as C
from utils import content
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds, mods = F["dataset"], F["models"]
best = L.best_model()
over = L.csv("overfitting.csv").set_index("model")
points = L.csv("roc_pr_points.csv")
METRICS = list(S.METRIC_LABELS)
T_LABEL = {"0.5": "Threshold 0.5", "cost-optimal": "Each model's cost-optimal threshold"}
N = len(L.MODEL_ORDER)

ui.header("Model comparison",
          "Every metric for every model, side by side: which model is best, on what, and why accuracy alone would "
          "have picked the wrong one. All values are from the holdout, scored once per model.", kicker="Models")

opt, m05 = L.metrics_at("cost-optimal"), L.metrics_at("0.5")
b = opt.loc[best]
ui.metrics_row([
    ("Best model", best, "Highest AUC, Gini, PR-AUC and KS; highest precision, recall and F1 at its cost-optimal threshold"),
    ("AUC", S.f4(b["AUC"]), content.METRIC_HELP["AUC"], f"{b['AUC'] - opt['AUC'].drop(best).max():+.4f} vs next best"),
    ("F1 (cost-optimal)", S.f4(b["F1"]), content.METRIC_HELP["F1"],
     f"{b['F1'] - opt['F1'].drop(best).max():+.4f} vs next best"),
    ("Saving vs approve-all", S.pct(b["saving_vs_approve_all"]), "Expected-cost saving under the assumed costs",
     f"{(b['saving_vs_approve_all'] - opt['saving_vs_approve_all'].drop(best).max()) * 100:+.2f} pp vs next best"),
])

# ---------------------------------------------------------------- 1. full table
st.subheader("1. Full metrics table")
t_type = st.segmented_control("Threshold", list(T_LABEL), default="cost-optimal", format_func=T_LABEL.get,
                              key="table_threshold") or "cost-optimal"
tbl = L.metrics_at(t_type)
show = tbl[["threshold"] + METRICS + ["CV_AUC_mean", "train_seconds"]].rename(
    columns={**S.METRIC_LABELS, "threshold": "Threshold", "CV_AUC_mean": "CV AUC", "train_seconds": "Train time (s)"})
better_high = [S.METRIC_LABELS[m] for m in METRICS if m != "LogLoss"] + ["CV AUC"]
styled = (show.style.format({c: "{:.4f}" for c in show.columns if c not in ("Threshold", "Train time (s)")} |
                            {"Threshold": "{:.2f}", "Train time (s)": "{:.1f}"})
          .highlight_max(subset=better_high, props="font-weight:700; background-color:rgba(42,120,214,0.18)")
          .highlight_min(subset=["Log loss", "Train time (s)"],
                         props="font-weight:700; background-color:rgba(42,120,214,0.18)"))
st.dataframe(styled)
winners = {S.METRIC_LABELS[m]: (tbl[m].idxmin() if m == "LogLoss" else tbl[m].idxmax()) for m in METRICS}
st.caption("Highlighted = best in the column (lowest for log loss and training time). Best per metric: " +
           ", ".join(f"{k}: {v}" for k, v in winners.items()) + ".")
ui.takeaway(f"{best} wins every threshold-free metric (AUC, Gini, PR-AUC, KS). At "
            f"{'the cost-optimal thresholds' if t_type == 'cost-optimal' else 'threshold 0.5'} the accuracy winner "
            f"is {winners['Accuracy']} and the F1 winner is {winners['F1']}: threshold-based metrics depend on how many "
            "applicants each model flags, which is why the threshold must be chosen deliberately.")

# ---------------------------------------------------------------- 2. metric explorer
st.subheader("2. Metric explorer")
c1, c2 = st.columns([2, 3])
metric = c1.selectbox("Metric", METRICS, index=METRICS.index("F1"), format_func=S.METRIC_LABELS.get)
t_exp = c2.segmented_control("Threshold ", list(T_LABEL), default="cost-optimal", format_func=T_LABEL.get,
                             key="explorer_threshold") or "cost-optimal"
vals = L.metrics_at(t_exp)[metric]
win = vals.idxmin() if metric == "LogLoss" else vals.idxmax()
lo = 0 if metric not in ("AUC",) else 0.5
ui.chart(C.model_bars(vals, ytitle=S.METRIC_LABELS[metric], best=win,
                      yrange=[lo, float(vals.max()) * 1.2]),
         title=f"{S.METRIC_LABELS[metric]} of the {N} models ({T_LABEL[t_exp].lower()})",
         subtitle=content.METRIC_HELP[metric])
st.caption(f"{S.METRIC_LABELS[metric]} is {'lower' if metric == 'LogLoss' else 'higher'} for better models.")
ui.takeaway(f"On {S.METRIC_LABELS[metric]} the best model is **{win}** ({vals[win]:.4f}); the spread between best and "
            f"worst is {abs(vals.max() - vals.min()):.4f}."
            + (" This metric does not depend on the threshold." if metric in S.THRESHOLD_FREE else
               " Switch the threshold to see how much this metric moves."))

# ---------------------------------------------------------------- 3. all metrics
st.subheader("3. All metrics at once")
t_all = st.segmented_control("Threshold  ", list(T_LABEL), default="cost-optimal", format_func=T_LABEL.get,
                             key="all_threshold") or "cost-optimal"
tbl_all = L.metrics_at(t_all)
tab1, tab2, tab3 = st.tabs(["Grouped bars", "Heatmap", "Radar"])
with tab1:
    bar_metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Gini", "PR_AUC", "KS"]
    ui.chart(C.grouped_bars(tbl_all, bar_metrics), title=f"{len(bar_metrics)} metrics, {N} models ({T_LABEL[t_all].lower()})",
             subtitle="Log loss is on its own scale (lower is better) and is shown in the table and heatmap",
             data=tbl_all[bar_metrics].reset_index(), formats={m: "{:.4f}" for m in bar_metrics})
    ui.takeaway(f"Ranking metrics barely move between models ({tbl_all['AUC'].min():.4f}–{tbl_all['AUC'].max():.4f} "
                f"AUC), while accuracy ({tbl_all['Accuracy'].min():.4f}–{tbl_all['Accuracy'].max():.4f}) and recall "
                f"({tbl_all['Recall'].min():.4f}–{tbl_all['Recall'].max():.4f}) swing with how many applicants are "
                "flagged.")
with tab2:
    labels = [S.METRIC_LABELS[m] for m in METRICS]
    values = tbl_all[METRICS].set_axis(labels, axis=1)
    norm = values.apply(lambda c: (c.max() - c) / (c.max() - c.min()) if c.name == "Log loss"
                        else (c - c.min()) / (c.max() - c.min()))
    ui.chart(C.metric_heatmap(values, norm), title="Models × metrics (colour = position in the column, darkest = best)",
             subtitle="Numbers are the actual metric values; log loss is inverted so that dark is always better",
             data=values.reset_index(), formats={c: "{:.4f}" for c in labels})
    ui.takeaway(f"{best}'s row is the darkest. Each column is scaled separately, so even small differences look "
                "large; read the numbers, not only the colours.")
with tab3:
    ui.chart(C.radar(norm), title=f"Radar of normalised metrics (1 = best of the {N}, 0 = worst)",
             subtitle="Each axis is scaled from the worst to the best model on that metric")
    ui.takeaway("A bigger, rounder shape means a model that is good on everything; spiky shapes reveal models that win "
                "one metric by giving up others (for example high accuracy bought with low recall).")

# ---------------------------------------------------------------- 4. ROC / PR
st.subheader("4. ROC and precision-recall curves")
tab1, tab2 = st.tabs(["ROC curves", "Precision-recall curves"])
with tab1:
    ui.chart(C.curves(points, "ROC"), title="ROC curves on the holdout",
             subtitle="Each line also has its own dash style; click legend entries to compare pairs")
    ui.takeaway(f"{best}'s curve lies on or above the others almost everywhere; the decision tree's curve is made of "
                "straight segments because a tree gives only one score per leaf.")
with tab2:
    ui.chart(C.curves(points, "PR", baseline=ds["holdout_default_rate"]), title="Precision-recall curves (default class)")
    ui.takeaway(f"With only {ds['holdout_default_rate']:.2%} defaulters, precision drops quickly as recall rises: even "
                f"the best model (PR-AUC {opt.loc[best, 'PR_AUC']:.4f}) must accept many false alarms to catch most "
                "defaulters.")

# ---------------------------------------------------------------- 5. confusion matrices
st.subheader("5. Confusion matrices side by side")
t_cm = st.segmented_control("Threshold   ", list(T_LABEL), default="cost-optimal", format_func=T_LABEL.get,
                            key="cm_threshold") or "cost-optimal"
tbl_cm = L.metrics_at(t_cm)
rows = [L.MODEL_ORDER[:3], L.MODEL_ORDER[3:]]
for row in rows:
    cols = st.columns(3)
    for col, name in zip(cols, row):
        r = tbl_cm.loc[name]
        with col:
            ui.chart(C.confusion(r["TN"], r["FP"], r["FN"], r["TP"], height=270),
                     title=f"{name} · threshold {r['threshold']:.2f}", key=f"cmp_cm_{name}_{t_cm}")
most_tp, fewest_fp = tbl_cm["TP"].idxmax(), tbl_cm["FP"].idxmin()
ui.takeaway(f"At {T_LABEL[t_cm].lower()}, {best} catches {int(tbl_cm.loc[best, 'TP']):,} of "
            f"{int(tbl_cm.loc[best, 'TP'] + tbl_cm.loc[best, 'FN']):,} defaulters"
            + (" (the most of any model)" if most_tp == best else
               f" ({most_tp} catches the most: {int(tbl_cm.loc[most_tp, 'TP']):,})")
            + f" and wrongly rejects {int(tbl_cm.loc[best, 'FP']):,} good applicants"
            + (" (the fewest)." if fewest_fp == best else
               f" ({fewest_fp} rejects the fewest: {int(tbl_cm.loc[fewest_fp, 'FP']):,}).")
            + " Every model trades the two errors differently.")

# ---------------------------------------------------------------- 6. threshold slider
st.subheader("6. Move the threshold yourself")
st.markdown("Every model gives each applicant a **risk score**; an applicant is rejected when the score is at or "
            "above the threshold. Drag the slider to see how accuracy, precision, recall and F1 change for every "
            "model (computed live from the holdout predictions).")
preds = L.parquet("holdout_predictions.parquet")
t = st.slider("Threshold", 0.05, 0.95, 0.50, 0.01)
live = pd.concat([metrics_by_threshold(preds["TARGET"], preds[f"p_{L.SHORT[m]}"], [t]).assign(model=m)
                  for m in L.MODEL_ORDER]).set_index("model")
ui.table(live[["Accuracy", "Precision", "Recall", "F1", "flagged_share", "TP", "FP", "FN", "TN"]].reset_index()
         .rename(columns={"flagged_share": "Rejected share"}),
         formats={"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
                  "Rejected share": "{:.2%}", "TP": "{:,}", "FP": "{:,}", "FN": "{:,}", "TN": "{:,}"})
pick = st.selectbox("Confusion matrix at this threshold for", L.MODEL_ORDER, index=L.MODEL_ORDER.index(best))
c1, c2 = st.columns([2, 3])
with c1:
    r = live.loc[pick]
    ui.chart(C.confusion(r["TN"], r["FP"], r["FN"], r["TP"], height=300), title=f"{pick} at threshold {t:.2f}",
             key="live_cm")
with c2:
    curves = L.parquet("threshold_curves.parquet")
    mt = st.segmented_control("Metric", ["Accuracy", "Precision", "Recall", "F1"], default="F1", key="curve_metric") or "F1"
    ui.chart(C.threshold_lines(curves, mt, marker=t, height=330), title=f"{mt} across thresholds 0.05–0.95",
             key="thr_lines")
f1_best = live["F1"].idxmax()
ui.takeaway(f"At threshold {t:.2f} the best F1 is {f1_best} ({live.loc[f1_best, 'F1']:.4f}); the models reject "
            f"between {live['flagged_share'].min():.1%} and {live['flagged_share'].max():.1%} of applicants. Raising "
            "the threshold rejects fewer people: recall falls and accuracy climbs towards the share of good applicants, "
            "which is exactly why accuracy rewards doing nothing.")

# ---------------------------------------------------------------- 7. overfitting
st.subheader("7. Overfitting check: train vs cross-validation vs holdout")
o = over.loc[L.MODEL_ORDER]
fig = go.Figure()
for col, label, color in [("train_AUC", "Train (in-sample)", S.series(6)), ("CV_AUC_mean", "5-fold CV", S.series(0)),
                          ("holdout_AUC", "Holdout", S.series(2))]:
    fig.add_trace(go.Bar(x=o.index, y=o[col], name=label, marker_color=color, marker_cornerradius=3,
                         text=[f"{v:.3f}" for v in o[col]], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}<br>" + label + " AUC %{y:.4f}<extra></extra>"))
C.layout(fig, ytitle="AUC", height=400, yrange=[0.5, 1.02])
ui.chart(fig, title="AUC on the training data, in cross-validation and on the holdout",
         data=o.reset_index(), formats={c: "{:+.4f}" if "minus" in c else "{:.4f}" for c in o.columns})
gap = o["train_minus_CV"]
ui.takeaway(f"Train − CV gaps: " + ", ".join(f"{m} {gap[m]:+.4f}" for m in L.MODEL_ORDER) + ". "
            f"Holdout AUC stays within {o['holdout_minus_CV'].abs().max():.4f} of CV for every model, so no model "
            "collapses on new applicants.")
st.markdown(f"""
- **High bias (underfitting):** logistic regression and the shallow decision tree have tiny gaps but low CV AUC
  ({o.loc['Logistic regression', 'CV_AUC_mean']:.4f} and {o.loc['Decision tree', 'CV_AUC_mean']:.4f}): they are too
  simple for the non-linear patterns.
- **High variance (overfitting):** fully grown trees memorise; the random forest scores
  {o.loc['Random forest', 'train_AUC']:.4f} on its own training data.
- **Bagging and boosting control variance:** averaging hundreds of trees keeps the forest's CV AUC at
  {o.loc['Random forest', 'CV_AUC_mean']:.4f}; regularised boosting with early stopping reaches the best CV AUC
  ({o.loc[best, 'CV_AUC_mean']:.4f} for {best}).
""")

# ---------------------------------------------------------------- 8. everything in one place
st.subheader("8. Everything in one place")
risk = {m: ("low" if gap[m] < 0.02 else "medium" if gap[m] < 0.1 else "high") for m in L.MODEL_ORDER}
ranked = opt["AUC"].sort_values(ascending=False).index.tolist()
verdict = {m: ("Selected: best ranking, best F1 and largest saving" if m == best else
               f"Ranks #{ranked.index(m) + 1} of {N} on holdout AUC") for m in L.MODEL_ORDER}
info = content.MODELS
rows = {
    "How it works": [info[m]["family"] for m in L.MODEL_ORDER],
    "Interpretability": [info[m]["interpretability"] for m in L.MODEL_ORDER],
    "Training time": [f"{opt.loc[m, 'train_seconds']:.1f} s" for m in L.MODEL_ORDER],
    "Needs scaling": [info[m]["scaling"] for m in L.MODEL_ORDER],
    "Handles missing values natively": [info[m]["missing"] for m in L.MODEL_ORDER],
    "Imbalance handling": [info[m]["imbalance"].split(":")[0] for m in L.MODEL_ORDER],
    "Overfitting risk (train − CV gap)": [f"{risk[m]} ({gap[m]:+.3f})" for m in L.MODEL_ORDER],
    "Holdout AUC": [f"{opt.loc[m, 'AUC']:.4f}" for m in L.MODEL_ORDER],
    "F1 at cost-optimal threshold": [f"{opt.loc[m, 'F1']:.4f}" for m in L.MODEL_ORDER],
    "Best use case": [info[m]["best_use"] for m in L.MODEL_ORDER],
    "Verdict": [verdict[m] for m in L.MODEL_ORDER],
}
st.dataframe(pd.DataFrame(rows, index=L.MODEL_ORDER).T, height=460)
st.caption("Overfitting risk bands: low below 0.02, medium below 0.10, high otherwise (train − CV AUC).")

# ---------------------------------------------------------------- 9. accuracy and verdict
st.subheader("9. Why accuracy is misleading here, and the final verdict")
acc_all = mods["approve_all_accuracy"]
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
**Approving everyone is {acc_all:.2%} accurate** on the holdout and catches **no** defaulter. Every real model is
*less* accurate than that at threshold 0.5 ({m05['Accuracy'].min():.2%}–{m05['Accuracy'].max():.2%}), because class
weighting makes them flag many applicants. The most "accurate" models are simply those that reject the fewest people:
at 0.5 that is the {m05['Accuracy'].idxmax().lower()} (recall {m05.loc[m05['Accuracy'].idxmax(), 'Recall']:.2%}),
at the cost-optimal thresholds the {opt['Accuracy'].idxmax().lower()}
(recall {opt.loc[opt['Accuracy'].idxmax(), 'Recall']:.2%}).

With about {ds['holdout_default_rate']:.0%} defaults, **AUC and PR-AUC** (ranking), **recall** (defaulters caught)
and **F1** (balance of the two errors) are what matter, plus the money: a missed defaulter costs far more than a
rejected good customer.
""")
with c2:
    second = ranked[1]
    st.markdown(f"""
**Verdict: {best}.**
- Best AUC ({opt.loc[best, 'AUC']:.4f}), Gini ({opt.loc[best, 'Gini']:.4f}), PR-AUC ({opt.loc[best, 'PR_AUC']:.4f})
  and KS ({opt.loc[best, 'KS']:.4f}); runner-up {second} (AUC {opt.loc[second, 'AUC']:.4f}).
- At its cost-optimal threshold ({opt.loc[best, 'threshold']:.2f}): best precision ({opt.loc[best, 'Precision']:.4f}),
  recall ({opt.loc[best, 'Recall']:.4f}) and F1 ({opt.loc[best, 'F1']:.4f}).
- Largest saving vs approving everyone ({opt.loc[best, 'saving_vs_approve_all']:.2%}), trains in
  {opt.loc[best, 'train_seconds']:.1f} s, handles missing values and categories natively, and every decision can be
  explained with SHAP reason codes (see *Thin-file & explainability*).
""")
ui.takeaway(f"Choosing by accuracy would have picked the model that rejects the fewest applicants. Choosing by "
            f"ranking quality, F1 and cost picks {best}, and it wins on all of them at once.")
