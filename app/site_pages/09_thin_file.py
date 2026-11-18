import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
eda, tf, sh = F["eda"], F["thin_file"], F["shap"]
groups = L.csv("thin_file_auc.csv").set_index("group")
abl = L.csv("ablation.csv").set_index("variant")
nb = groups.loc["Bureau history: no"]

ui.header("Thin-file applicants and explainability",
          "How well the final LightGBM works for applicants **without a Credit Bureau record**, what the model relies "
          "on, and why it rejects a given applicant.", kicker="Decisions")

st.subheader("What 'thin file' means here, and a caveat")
ui.metrics_row([
    ("No Credit Bureau record", S.num(eda["no_bureau"]), "Labelled applicants with no row in bureau.csv"),
    ("…of whom have Home Credit history", S.pct(eda["no_bureau_with_prev_share"]),
     "They applied to Home Credit before, so the lender's own data is available"),
    ("Truly new to credit", S.num(eda["neither"]), "No bureau record AND no previous Home Credit application"),
    ("…in the holdout", f"{tf['neither_holdout']:,} ({tf['neither_holdout_defaults']} defaults)",
     "Too few to evaluate reliably"),
])
ui.takeaway(f"Most 'thin-file' applicants are not new to credit: {eda['no_bureau_with_prev_share']:.2%} of those without "
            f"a bureau record have earlier Home Credit applications. Only {eda['neither']:,} labelled applicants have "
            f"no history at all; in the holdout ({tf['neither_holdout']:,} applicants) LightGBM's AUC is "
            f"{tf['neither_auc']:.4f}, but its 95% bootstrap interval ({tf['neither_ci'][0]:.4f}–{tf['neither_ci'][1]:.4f}) "
            "is far too wide for any conclusion about genuinely new customers.")

st.subheader("1. Accuracy by credit-history group")
order = list(groups.index)
fig = go.Figure()
colors = S.model_colors()
for m in L.MODEL_ORDER:
    col = f"{m} AUC"
    err = None
    if f"{m} CI low" in groups:
        err = dict(type="data", symmetric=False, array=groups[f"{m} CI high"] - groups[col],
                   arrayminus=groups[col] - groups[f"{m} CI low"], thickness=1.2, width=3)
    fig.add_trace(go.Bar(x=order, y=groups[col], name=m, marker_color=colors[m], marker_cornerradius=3, error_y=err,
                         hovertemplate=f"<b>{m}</b><br>%{{x}}<br>AUC %{{y:.4f}}<extra></extra>"))
C.layout(fig, ytitle="Holdout AUC", height=430, yrange=[0.5, 0.86])
with_ci = [m for m in L.MODEL_ORDER if f"{m} CI low" in groups]
ui.chart(fig, title=f"Holdout AUC by group (95% bootstrap intervals, {tf['n_boot']} resamples, for "
                    f"{', '.join(with_ci)}, as in notebook 04)",
         data=groups.reset_index(), formats={c: "{:.4f}" for c in groups.columns if "AUC" in c or "CI" in c} |
         {"applicants": "{:,}", "default_rate": "{:.2%}"})
ui.takeaway(f"For the {int(nb['applicants']):,} holdout applicants without a bureau record (default rate "
            f"{nb['default_rate']:.2%}), LightGBM still reaches AUC {nb['LightGBM AUC']:.4f} "
            f"({nb['LightGBM CI low']:.4f}–{nb['LightGBM CI high']:.4f}) against "
            f"{groups.loc['Bureau history: yes', 'LightGBM AUC']:.4f} for those with one. Logistic regression loses "
            f"more on this group ({nb['Logistic regression AUC']:.4f}).")

st.subheader("2. Ablation: what if bureau data and external scores were not available?")
fig = go.Figure()
for col, lo, hi, name, color in [("AUC all", "all CI low", "all CI high", "All holdout applicants", S.series(0)),
                                 ("AUC no-bureau", "no-bureau CI low", "no-bureau CI high",
                                  "Applicants without a bureau record", S.series(1))]:
    fig.add_trace(go.Bar(x=abl.index, y=abl[col], name=name, marker_color=color, marker_cornerradius=3,
                         error_y=dict(type="data", symmetric=False, array=abl[hi] - abl[col],
                                      arrayminus=abl[col] - abl[lo], thickness=1.2, width=3),
                         text=[f"{v:.3f}" for v in abl[col]], textposition="inside",
                         hovertemplate=f"{name}<br>%{{x}}<br>AUC %{{y:.4f}}<extra></extra>"))
C.layout(fig, ytitle="Holdout AUC (LightGBM, same tuned settings)", height=420, yrange=[0.5, 0.86])
ui.chart(fig, title="LightGBM refitted without groups of features",
         data=abl.reset_index(), formats={c: "{:.4f}" for c in abl.columns if c != "n_features"})
alt = abl.loc["Alternative data only"]
ui.takeaway(f"Removing the external scores costs the most ({abl.loc['Full model', 'AUC all']:.4f} → "
            f"{abl.loc['No EXT_SOURCE', 'AUC all']:.4f}). With **alternative data only** (no bureau data, no external "
            f"scores; {int(alt['n_features'])} features) the model still ranks bureau-less applicants with AUC "
            f"{alt['AUC no-bureau']:.4f}, only {abl.loc['Full model', 'AUC no-bureau'] - alt['AUC no-bureau']:.4f} below "
            "the full model on that group: the application form and the lender's own history carry real signal.")

st.subheader("3. What drives the model's scores (SHAP)")
imp = L.csv("shap_importance.csv").head(20).iloc[::-1]
fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker_color=S.series(0),
                       marker_cornerradius=3, text=[f"{v:.3f}" for v in imp["mean_abs_shap"]], textposition="outside",
                       cliponaxis=False, hovertemplate="<b>%{y}</b><br>mean |SHAP| %{x:.4f}<extra></extra>"))
C.layout(fig, xtitle="Mean |SHAP value| (log-odds of default)", height=620, legend=False,
         xrange=[0, imp["mean_abs_shap"].max() * 1.2])
ui.chart(fig, title=f"Global importance: top 20 features by mean |SHAP| ({sh['sample']:,} holdout applicants)",
         data=imp.iloc[::-1], formats={"mean_abs_shap": "{:.4f}"})
top = imp.iloc[::-1]
ui.takeaway(f"**{top['feature'].iloc[0]}** dominates (mean |SHAP| {top['mean_abs_shap'].iloc[0]:.4f}, "
            f"{top['mean_abs_shap'].iloc[0] / top['mean_abs_shap'].iloc[1]:.2f}× the next feature). Then come loan "
            f"structure ({top['feature'].iloc[1]}, {top['feature'].iloc[3]}), the lender's own history "
            f"({top['feature'].iloc[2]}) and, worryingly, gender (see the Fairness page).")

st.markdown("**Direction of the top drivers** — average SHAP value by decile of the feature (D1 = lowest value)")
dirn = L.csv("shap_direction.csv").set_index("feature")
deciles = [f"D{i}" for i in range(1, 11)]
ui.chart(C.diverging_heatmap(dirn[deciles].to_numpy(), deciles, list(dirn.index), height=360,
                             colorbar_title="mean SHAP"),
         subtitle="Red = pushes towards rejection, blue = towards approval",
         data=dirn.reset_index(), formats={c: "{:+.3f}" for c in dirn.columns})
trend = ["lower value → higher risk" if r < -0.5 else "higher value → higher risk" if r > 0.5 else "not monotonic"
         for r in dirn["spearman"]]
st.markdown("\n".join(f"- `{f}`: {t} (rank correlation of value and SHAP {r:+.2f})"
                      for f, t, r in zip(dirn.index, trend, dirn["spearman"])))
gm = sh["gender_mean_shap"]
ui.takeaway(f"Low external scores, low goods-to-credit ratios and a low requested-to-granted ratio on earlier "
            "applications raise risk; CREDIT_TERM is not monotonic. Gender also shifts scores (mean SHAP "
            f"{gm.get('M', 0):+.3f} for men vs {gm.get('F', 0):+.3f} for women).")
with st.expander("SHAP beeswarm (static image from notebook 04)", icon=":material/image:"):
    st.image(str(L.figure("26_shap_beeswarm.png")), width="stretch")

st.subheader("4. Why were these applicants rejected? (reason codes)")
rc = L.csv("reason_codes.csv")
wf = L.csv("reason_waterfalls.csv")
st.caption(f"{rc['applicant'].nunique()} holdout applicants rejected at the cost-optimal threshold "
           f"({F['models']['thresholds']['LightGBM']:.2f}) "
           f"from the SHAP sample ({sh['rejected_in_sample']:,} of {sh['sample']:,} were rejected): the highest-risk "
           "one, the median rejected one and the one closest to the threshold. No identifiers are shown.")
tabs = st.tabs([f"Applicant {k}: {g['label'].iloc[0]}" for k, g in rc.groupby("applicant")])
for tab, (k, g) in zip(tabs, rc.groupby("applicant")):
    with tab:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.metric("Risk score", f"{g['score'].iloc[0]:.3f}", border=True)
            st.markdown("**Actual outcome:** " + ("defaulted" if g["actual"].iloc[0] == 1 else "repaid (a good "
                                                                                              "customer the policy loses)"))
            st.markdown("**Top reasons for rejection**\n" +
                        "\n".join(f"{i}. {r} *(+{s:.3f} log-odds)*" for i, (r, s) in
                                  enumerate(zip(g["reason"], g["shap_log_odds"]), 1)))
        with c2:
            w = wf[wf["applicant"] == k]
            labels = [f"{f} = {v}" if isinstance(v, str) and v else f for f, v in zip(w["feature"], w["value"].fillna(""))]
            ui.chart(C.waterfall(labels, w["shap"], sh["base_value"]), title="How each feature moved the score",
                     key=f"wf_{k}")
outcomes = rc.groupby("applicant")["actual"].first()
n_repaid = int((outcomes == 0).sum())
ui.takeaway("Each rejection can be explained in plain English from the features that pushed the score up most. "
            f"{n_repaid} of the {len(outcomes)} example applicants actually repaid: rejecting some good customers is "
            "the price a cost-optimal policy pays to avoid larger losses.")
