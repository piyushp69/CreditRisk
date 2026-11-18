import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
fa = F["fairness"]
fair = L.csv("fairness.csv")
proxy = L.csv("proxy_features.csv")
WITH, WITHOUT = "with CODE_GENDER", "without CODE_GENDER"

ui.header("Fairness", "Are approvals and mistakes spread fairly across gender and age? Checked for the final LightGBM "
          "at its cost-optimal threshold, and for the same model retrained **without** gender.", kicker="Decisions")

version = st.segmented_control("Model version", [WITH, WITHOUT], default=WITH,
                               format_func=lambda v: f"LightGBM {v} (threshold "
                                                     f"{fa['t_with'] if v == WITH else fa['t_without']:.2f})") or WITH
for attribute in ["Gender", "Age group"]:
    d = fair[(fair["model"] == version) & (fair["attribute"] == attribute)].reset_index(drop=True)
    st.subheader(f"By {attribute.lower()}")
    fig = go.Figure()
    for col, name, color in [("approval_rate", "Approval rate", S.series(0)),
                             ("actual_default_rate", "Actual default rate", S.series(1)),
                             ("good_applicants_rejected", "Good applicants rejected", S.series(2))]:
        fig.add_trace(go.Bar(x=d["group"].astype(str), y=d[col], name=name, marker_color=color, marker_cornerradius=3,
                             text=[f"{v:.1%}" for v in d[col]], textposition="outside", cliponaxis=False,
                             hovertemplate=f"{name}<br>%{{x}}: %{{y:.2%}}<extra></extra>"))
    C.layout(fig, ytitle="Share", height=400, ypct=True, yrange=[0, 1.12])
    ui.chart(fig, title=f"Approval, actual default and wrongful rejection by {attribute.lower()} ({version})",
             data=d.drop(columns=["model", "attribute"]),
             formats={"applicants": "{:,}", "actual_default_rate": "{:.2%}", "approval_rate": "{:.2%}",
                      "default_rate_approved": "{:.2%}", "good_applicants_rejected": "{:.2%}", "threshold": "{:.2f}"},
             key=f"fair_{attribute}_{version}")
    hi = d.loc[d["good_applicants_rejected"].idxmax()]
    lo = d.loc[d["good_applicants_rejected"].idxmin()]
    ui.takeaway(f"Approval ranges from {d['approval_rate'].min():.2%} to {d['approval_rate'].max():.2%}, in line with "
                f"actual default rates ({d['actual_default_rate'].min():.2%}–{d['actual_default_rate'].max():.2%}). "
                f"But good applicants who would have repaid are rejected {hi['good_applicants_rejected'] / lo['good_applicants_rejected']:.2f}× "
                f"as often in group {hi['group']} ({hi['good_applicants_rejected']:.2%}) as in group {lo['group']} "
                f"({lo['good_applicants_rejected']:.2%}): the cost of the model's mistakes is not shared equally.")

st.subheader("With vs without CODE_GENDER")
g = fair[fair["attribute"] == "Gender"].set_index(["model", "group"])
gap_with = g.loc[(WITH, "F"), "approval_rate"] - g.loc[(WITH, "M"), "approval_rate"]
gap_without = g.loc[(WITHOUT, "F"), "approval_rate"] - g.loc[(WITHOUT, "M"), "approval_rate"]
ui.metrics_row([
    ("Holdout AUC without gender", S.f4(fa["auc_without"]), "Same settings, CODE_GENDER removed",
     f"{fa['auc_without'] - fa['auc_with']:+.4f}"),
    ("Approval gap F − M", f"{gap_without * 100:.2f} pp", "Difference in approval rates without gender",
     f"{(gap_without - gap_with) * 100:+.2f} pp"),
    ("Threshold (own OOF optimum)", f"{fa['t_without']:.2f}", f"With gender: {fa['t_with']:.2f}"),
    ("Holdout cost change", f"{(fa['cost_without'] / fa['cost_with'] - 1):+.2%}",
     f"{fa['cost_without'] - fa['cost_with']:+,.0f} currency units"),
])
ui.takeaway(f"Dropping gender costs almost nothing in accuracy (AUC {fa['auc_with']:.4f} → {fa['auc_without']:.4f}) "
            f"and narrows the approval gap from {gap_with * 100:.2f} to {gap_without * 100:.2f} percentage points, but "
            "does not close it.")

st.subheader("Why the gap does not disappear: proxy discrimination")
c1, c2 = st.columns([2, 3])
with c1:
    st.markdown(f"""
A model trained to predict **gender** from the remaining features reaches **AUC {fa['proxy_auc']:.4f}** on the
holdout. Other inputs (occupation above all) act as *proxies* for gender, so removing the column ("fairness through
unawareness") only partly removes its influence. Part of the gap also reflects a real difference in repayment:
men's actual default rate is {g.loc[(WITH, 'M'), 'actual_default_rate']:.2%} against
{g.loc[(WITH, 'F'), 'actual_default_rate']:.2%} for women.
""")
with c2:
    p = proxy.iloc[::-1]
    fig = go.Figure(go.Bar(x=p["gain_share"], y=p["feature"], orientation="h", marker_color=S.series(6),
                           marker_cornerradius=3, text=[f"{v:.1%}" for v in p["gain_share"]], textposition="outside",
                           cliponaxis=False, hovertemplate="<b>%{y}</b><br>%{x:.2%} of gain<extra></extra>"))
    C.layout(fig, xtitle="Share of the gender model's gain", height=360, legend=False, xpct=True,
             xrange=[0, p["gain_share"].max() * 1.25])
    ui.chart(fig, title="Features that reveal gender most", data=proxy, formats={"gain_share": "{:.2%}"})
ui.takeaway(f"{proxy['feature'].iloc[0]} alone carries {proxy['gain_share'].iloc[0]:.1%} of the gender model's "
            "gain. Any model that uses it can treat men and women differently even without the gender column.")

st.subheader("Decision taken")
st.markdown(f"""
1. **Remove `CODE_GENDER`** from the production model: AUC changes by {fa['auc_without'] - fa['auc_with']:+.4f} and
   holdout cost by {(fa['cost_without'] / fa['cost_with'] - 1):+.2%}, a small price for not using a protected attribute
   that many credit laws forbid (e.g. US ECOA / Regulation B, EU Directive 2004/113/EC).
2. **Monitor outcomes, not just inputs:** track approval rates and the share of good applicants rejected by gender and
   age after deployment (an *equal-opportunity* check), because proxies keep part of the gap.
3. **Keep age under review:** younger applicants really do default more, but they also bear more wrongful rejections;
   a lender must check what its local law allows.
""")
