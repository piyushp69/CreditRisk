import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
cl = F["clustering"]
K = cl["chosen_k"]
ks = L.csv("kmeans_k_selection.csv")
prof = L.csv("cluster_profiles.csv")          # sorted by training default rate
centres = L.csv("cluster_centres.csv").set_index("cluster")
exp = L.csv("cluster_feature_experiment.csv").set_index("model")
FEATS = ["AGE_YEARS", "LOG_INCOME", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "YEARS_EMPLOYED", "EXT_SOURCE_MEAN",
         "INST_LATE_SHARE", "PREV_REFUSED_SHARE"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "star", "cross", "pentagon"]

ui.header("Clustering: applicant risk segments",
          f"K-Means groups the training applicants into **{K} segments** using {len(FEATS)} profile features, "
          "without looking at who defaulted. Only afterwards do we check how often each segment defaults.",
          kicker="Analysis")
with st.expander("How K-Means works and how the data were prepared", icon=":material/info:"):
    st.markdown(f"""
**K-Means** picks *k* centres, assigns every applicant to the nearest centre, moves each centre to the average of its
applicants, and repeats until nothing changes. It minimises the total squared distance to the centres (*inertia*).

**Features:** {", ".join(f"`{f}`" for f in FEATS)} (income as its logarithm because it is very skewed).
**Preparation (fitted on the training set only):** median imputation for gaps (pensioners have no employment
length, first-time customers no repayment history) and standardisation, because K-Means measures distances and a
feature in large units would otherwise dominate. Holdout applicants are only assigned to the fitted centres.
""")

# ---------------------------------------------------------------- choosing k
st.subheader("1. Choosing the number of clusters")
c1, c2 = st.columns(2)
with c1:
    ui.chart(C.lines(ks["k"], [("Inertia", ks["inertia"], S.series(0))], "Number of clusters k",
                     "Within-cluster sum of squares", marker_x=K, marker_label=f"chosen k = {K}", yfmt=",.0f"),
             title="Elbow method", data=ks[["k", "inertia", "inertia_drop"]],
             formats={"inertia": "{:,.0f}", "inertia_drop": "{:,.0f}"})
with c2:
    ui.chart(C.lines(ks["k"], [("Silhouette", ks["silhouette_20k_sample"], S.series(1))], "Number of clusters k",
                     "Mean silhouette (higher = better separated)", marker_x=K, marker_label=f"chosen k = {K}",
                     yfmt=".4f"),
             title=f"Silhouette score ({cl['silhouette_sample_size']:,}-applicant sample)",
             data=ks[["k", "silhouette_20k_sample", "default_rate_min", "default_rate_max"]],
             formats={"silhouette_20k_sample": "{:.4f}", "default_rate_min": "{:.2%}", "default_rate_max": "{:.2%}"})
k_tab = ks.set_index("k")
ui.takeaway(f"The silhouette is highest at k = 2 ({k_tab.loc[2, 'silhouette_20k_sample']:.4f}), but those two halves "
            f"default at {k_tab.loc[2, 'default_rate_min']:.2%} and {k_tab.loc[2, 'default_rate_max']:.2%}, too "
            f"similar to be useful. Among k ≥ 3 the best silhouette is at **k = {K}** "
            f"({k_tab.loc[K, 'silhouette_20k_sample']:.4f}), which is also where the elbow bends: the {K}th cluster "
            f"still removes {k_tab.loc[K, 'inertia_drop']:,.0f} of inertia, the next one only "
            f"{k_tab.loc[K + 1, 'inertia_drop']:,.0f}. All silhouettes are low, so the segments overlap rather than "
            "form sharply separated groups.")

# ---------------------------------------------------------------- segment cards
st.subheader("2. The segments")
cols = st.columns(3)
for i, (_, r) in enumerate(prof.iterrows()):
    with cols[i % 3].container(border=True):
        level = r["risk_level"]
        icon = {"low": ":material/check_circle:", "medium": ":material/remove_circle:", "high": ":material/error:"}[level]
        st.markdown(f"**{r['segment'].rsplit(' (', 1)[0]}**")
        st.caption(f"{icon} {level.capitalize()} risk · {r['applicants']:,} applicants ({r['share']:.1%})")
        st.metric("Default rate (train)", S.pct(r["default_rate"]),
                  delta=f"{(r['default_rate'] - cl['overall_train_default_rate']) * 100:+.2f} pp vs overall",
                  delta_color="inverse")
        st.markdown(f"Age {r['mean_AGE_YEARS']:.1f} · ext. score {r['mean_EXT_SOURCE_MEAN']:.3f}  \n"
                    f"Loan / income {r['mean_CREDIT_INCOME_RATIO']:.2f} · years employed {r['mean_YEARS_EMPLOYED']:.1f}  \n"
                    f"Late instalments {r['mean_INST_LATE_SHARE']:.1%} · refused {r['mean_PREV_REFUSED_SHARE']:.1%}  \n"
                    f"Holdout default rate {r['holdout_default_rate']:.2%}")
st.caption("Profile values are segment means in original units (years employed excludes pensioners). Names come "
           "from each segment's most extreme standardised centre; risk level compares the default rate with the "
           "overall rate (low below 0.8×, high above 1.2×).")

colors = [S.series(i) for i in range(K)]
fig = go.Figure()
for name, col, color in [("Train", "default_rate", S.series(1)), ("Holdout", "holdout_default_rate", S.series(2))]:
    fig.add_trace(go.Bar(y=prof["segment"], x=prof[col], name=name, orientation="h", marker_color=color,
                         marker_cornerradius=3, text=[f"{v:.1%}" for v in prof[col]], textposition="outside",
                         cliponaxis=False, hovertemplate="%{y}<br>" + name + " default rate %{x:.2%}<extra></extra>"))
fig.add_vline(x=cl["overall_train_default_rate"], line=dict(color=S.muted(), width=1, dash="dash"),
              annotation_text=f"overall {cl['overall_train_default_rate']:.2%}", annotation_position="bottom right")
C.layout(fig, xtitle="Default rate", height=430, xpct=True, xrange=[0, prof["default_rate"].max() * 1.3])
fig.update_yaxes(autorange="reversed")
ui.chart(fig, title="Default rate by segment: training applicants vs holdout applicants",
         data=prof[["segment", "applicants", "share", "default_rate", "holdout_applicants", "holdout_default_rate"]],
         formats={"applicants": "{:,}", "share": "{:.1%}", "default_rate": "{:.2%}", "holdout_applicants": "{:,}",
                  "holdout_default_rate": "{:.2%}"})
gap = (prof["holdout_default_rate"] - prof["default_rate"]).abs().max()
ui.takeaway(f"Default rates range from {prof['default_rate'].min():.2%} (**{prof['segment'].iloc[0]}**) to "
            f"{prof['default_rate'].max():.2%} (**{prof['segment'].iloc[-1]}**). Holdout applicants assigned to the "
            f"same centres differ by at most {gap * 100:.2f} percentage points, so the segments generalise.")

# ---------------------------------------------------------------- heatmap
st.subheader("3. What defines each segment")
z = centres.loc[prof["cluster"], FEATS].to_numpy()
ui.chart(C.diverging_heatmap(z, FEATS, list(prof["segment"]), height=420),
         title="Cluster centres in standard deviations from the training average",
         subtitle="Red = above average, blue = below average",
         data=centres.set_index("segment")[FEATS].reset_index(), formats={f: "{:+.2f}" for f in FEATS})
strongest = {prof.loc[i, "segment"]: FEATS[int(np.argmax(np.abs(z[i])))] for i in range(K)}
ui.takeaway("Each segment is defined by one or two features: " +
            "; ".join(f"{seg.rsplit(' (', 1)[0]} → `{f}`" for seg, f in strongest.items()) +
            ". The riskiest segments are defined by behaviour with the lender (late instalments, refusals), the safest "
            "by age, tenure and strong external scores.")

# ---------------------------------------------------------------- scatter
st.subheader("4. The segments in two dimensions")
sample = L.parquet("cluster_plot_sample.parquet")
f1, f2 = cl["most_separating_features"]
view = st.segmented_control("View", ["2-component PCA (plotting only)", f"{f1} vs {f2}"],
                            default="2-component PCA (plotting only)")
pca_view = view != f"{f1} vs {f2}"
fig = go.Figure()
for i, (_, r) in enumerate(prof.iterrows()):
    d = sample[sample["cluster"] == r["cluster"]]
    if pca_view:
        x, y = d["x"], d["y"]
    else:
        x = (d[f1] - sample[f1].mean()) / sample[f1].std()
        y = (d[f2] - sample[f2].mean()) / sample[f2].std()
    fig.add_trace(go.Scattergl(x=x, y=y, mode="markers", name=r["segment"],
                               marker=dict(size=5, color=colors[i], symbol=SYMBOLS[i], opacity=0.55),
                               hovertemplate=r["segment"] + "<extra></extra>"))
pv = cl["plot_pca_variance"]
C.layout(fig, xtitle=f"Plotting component 1 ({pv[0]:.1%} of variance)" if pca_view else f"{f1} (standardised)",
         ytitle=f"Plotting component 2 ({pv[1]:.1%} of variance)" if pca_view else f"{f2} (standardised)",
         height=540, legend_bottom=True)
ui.chart(fig, title=f"{cl['plot_sample_size']:,} random training applicants coloured by segment",
         subtitle="Each segment also has its own marker shape. Click legend entries to hide segments.")
ui.takeaway(f"The PCA view shows {sum(pv):.1%} of the variation: it separates the loan-to-income and the age / tenure "
            f"segments, while the behavioural segments overlap there. Switching to `{f1}` and `{f2}` (the features "
            "that separate the clusters most) pulls the late payers and the often-refused applicants apart.")

# ---------------------------------------------------------------- experiment
st.subheader("5. Does the segment label help the model?")
base, with_c = exp.iloc[0], exp.iloc[1]
ui.metrics_row([
    ("LightGBM CV AUC", S.f4(base["CV_AUC_mean"]), "5-fold CV on train, without the segment label"),
    ("+ segment label", S.f4(with_c["CV_AUC_mean"]), "Same model with KMEANS_CLUSTER added",
     f"{with_c['CV_AUC_mean'] - base['CV_AUC_mean']:+.4f}"),
    ("LightGBM holdout AUC", S.f4(base["holdout_AUC"]), "Without the segment label"),
    ("+ segment label", S.f4(with_c["holdout_AUC"]), "Holdout, evaluated once",
     f"{with_c['holdout_AUC'] - base['holdout_AUC']:+.4f}"),
])
ui.takeaway(f"Adding the segment label changes holdout AUC by {cl['holdout_auc_change']:+.4f}: no real gain. The "
            "trees already see the raw features the segments are built from, so the label only repackages that "
            "information. Segments are valuable for *describing* risk groups (marketing, limits, collections), not for "
            "improving the score.")
