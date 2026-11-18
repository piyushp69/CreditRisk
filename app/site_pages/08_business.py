import plotly.graph_objects as go
import streamlit as st

from src.cost import approve_all_cost, decision_summary, optimal_threshold, threshold_sweep
from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
cfg, ds = F["config"], F["dataset"]
best = L.best_model()

ui.header("Business decision: where to set the cut-off",
          "A model only gives each applicant a **risk score**. The lender still has to choose a threshold: reject if "
          "the score is at or above it. The best threshold depends on what each kind of mistake costs.",
          kicker="Decisions")
st.warning("Scores are **risk scores, not probabilities**: class weighting pushes them above real default rates, so "
           "a score of 0.7 does not mean a 70% chance of default. The costs below are **assumptions** you can change.",
           icon=":material/info:")

st.subheader("Your assumptions")
c1, c2, c3 = st.columns(3)
model = c1.selectbox("Model", L.MODEL_ORDER, index=L.MODEL_ORDER.index(best))
lgd = c2.slider("Loss when a defaulter is approved (LGD, share of the loan)", 0.20, 0.90, float(cfg["lgd"]), 0.05,
                help="Approving an applicant who defaults costs LGD × loan amount.")
margin = c3.slider("Profit lost when a good applicant is rejected (share of the loan)", 0.05, 0.30,
                   float(cfg["margin"]), 0.01, help="Rejecting an applicant who would have repaid costs margin × loan amount.")
lgd, margin = round(lgd, 2), round(margin, 2)

grid = L.csv("threshold_grid.csv")
row = grid[(grid["model"] == model) & grid["LGD"].round(2).eq(lgd) & grid["MARGIN"].round(2).eq(margin)]
t_train = float(row["optimal_threshold"].iloc[0])


@st.cache_data(show_spinner=False)
def sweep(model_name, lgd_, margin_):
    p = L.parquet("holdout_predictions.parquet")
    s = threshold_sweep(p["TARGET"], p[f"p_{L.SHORT[model_name]}"], p["AMT_CREDIT"], lgd=lgd_, margin=margin_)
    return s, approve_all_cost(p["TARGET"], p["AMT_CREDIT"], lgd_)


sw, base = sweep(model, lgd, margin)
manual = st.toggle("Set the threshold myself instead", value=False)
t_used = st.slider("Threshold", 0.01, 0.99, t_train, 0.01) if manual else t_train
s = sw.loc[(sw["threshold"] - t_used).abs().idxmin()]
s05 = sw.loc[(sw["threshold"] - 0.5).abs().idxmin()]
t_holdout = optimal_threshold(sw)

st.subheader("Result on the holdout applicants")
ui.metrics_row([
    ("Threshold used", f"{t_used:.2f}", "Chosen on the training out-of-fold predictions for these costs"
     if not manual else "Your manual choice"),
    ("Saving vs approve everyone", S.pct(1 - s["total_cost"] / base),
     "Expected cost compared with approving every applicant"),
    ("Approval rate", S.pct(s["approval_rate"]), "Share of applicants approved"),
    ("Default rate among approved", S.pct(s["default_rate_approved"]),
     f"Approving everyone: {ds['holdout_default_rate']:.2%}"),
])
ui.metrics_row([
    ("Total expected cost", f"{s['total_cost'] / 1e6:,.1f} M", "Loss on approved defaulters + lost margin (currency units)"),
    ("Approve-everyone cost", f"{base / 1e6:,.1f} M", "Every defaulter is approved"),
    ("Defaulters approved", S.num(s["n_fn"]), "False negatives"),
    ("Good applicants rejected", S.num(s["n_fp"]), "False positives"),
])

fig = go.Figure()
for col, name, color in [("total_cost", "Total cost", S.series(0)),
                         ("fn_cost", "Loss from approved defaulters", S.series(1)),
                         ("fp_cost", "Lost margin on rejected good applicants", S.series(2))]:
    fig.add_trace(go.Scatter(x=sw["threshold"], y=sw[col] / 1e6, mode="lines", name=name,
                             line=dict(color=color, width=3 if col == "total_cost" else 2),
                             hovertemplate=f"{name}<br>threshold %{{x:.2f}}<br>%{{y:,.1f}} M<extra></extra>"))
fig.add_hline(y=base / 1e6, line=dict(color=S.muted(), width=1, dash="dot"),
              annotation_text="approve everyone", annotation_position="top left")
fig.add_vline(x=t_used, line=dict(color=S.STATUS["critical"], width=1.5),
              annotation_text=f"used {t_used:.2f}", annotation_position="bottom left")
fig.add_vline(x=0.5, line=dict(color=S.muted(), width=1, dash="dash"), annotation_text="0.5",
              annotation_position="bottom right")
C.layout(fig, xtitle="Threshold (reject if score ≥ threshold)", ytitle="Cost (millions of currency units)",
         height=440, hovermode="x unified", xrange=[0, 1])
ui.chart(fig, title=f"Expected cost vs threshold: {model}, LGD {lgd:.2f}, margin {margin:.2f}",
         data=sw.assign(saving=1 - sw["total_cost"] / base),
         formats={"threshold": "{:.2f}", "total_cost": "{:,.0f}", "fn_cost": "{:,.0f}", "fp_cost": "{:,.0f}",
                  "approval_rate": "{:.2%}", "default_rate_approved": "{:.2%}", "saving": "{:+.2%}"})
ui.takeaway(f"Low thresholds reject too many good applicants (the lost-margin line rises); high thresholds approve "
            f"too many defaulters (the loss line rises). For these costs the training data pick {t_train:.2f}; on the holdout the "
            f"cheapest threshold would have been {t_holdout:.2f} (shown for reference only, never used to choose). "
            f"The conventional 0.5 gives a saving of {1 - s05['total_cost'] / base:+.2%} for {model}.")

st.subheader("How the best threshold moves with the costs")
gm = grid[grid["model"] == model].pivot(index="LGD", columns="MARGIN", values="optimal_threshold")
heat = go.Figure(go.Heatmap(z=gm.to_numpy(), x=[f"{m:.2f}" for m in gm.columns], y=[f"{l:.2f}" for l in gm.index],
                            colorscale=S.sequential_scale(), colorbar=dict(title=dict(text="threshold"), thickness=12),
                            hovertemplate="LGD %{y}, margin %{x}<br>best threshold %{z:.2f}<extra></extra>"))
heat.add_trace(go.Scatter(x=[f"{margin:.2f}"], y=[f"{lgd:.2f}"], mode="markers",
                          marker=dict(size=14, symbol="circle-open", color=S.STATUS["critical"], line=dict(width=3)),
                          name="your choice", hoverinfo="skip"))
C.layout(heat, xtitle="Lost margin (share of the loan)", ytitle="Loss given default (share of the loan)", height=460,
         legend=False)
ui.chart(heat, title=f"Cost-optimal threshold for every LGD × margin pair ({model}, chosen on training data)",
         data=gm.set_axis([f"margin {c:.2f}" for c in gm.columns], axis=1).reset_index(),
         formats={f"margin {c:.2f}": "{:.2f}" for c in gm.columns} | {"LGD": "{:.2f}"})
ui.takeaway("A bigger loss on defaulters pushes the threshold down (reject more); a bigger lost margin pushes it up "
            "(approve more). Only the ratio of the two costs matters, so the diagonal bands share one threshold.")
with st.expander("How the cost is calculated", icon=":material/calculate:"):
    st.markdown(f"""
For every holdout applicant with loan amount *A*:
- approved and defaults → cost = LGD × *A* (loss given default);
- rejected but would have repaid → cost = margin × *A* (lost profit);
- correct decisions cost nothing.

The total over all {ds['n_holdout']:,} holdout applicants is compared with approving everyone. To avoid choosing on the
data used for reporting, the threshold for each (LGD, margin) pair was picked on the **training out-of-fold
predictions** of the selected model (precomputed), then applied to the holdout. Amounts are in the data's undisclosed
currency.
""")
