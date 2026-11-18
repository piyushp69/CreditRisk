import numpy as np
import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui
from utils.predictor import FORM_CATEGORIES, FORM_FIELDS, build_features, form_values, reasons, score

meta = L.predictor_meta()
features, categories, defaults = meta["features"], meta["categories"], meta["defaults"]
threshold = meta["threshold"]
TYPICAL = "Typical applicant (training medians / most common values)"

ui.header("Live loan-risk predictor",
          "Describe an applicant and the final LightGBM scores them, applies the cost-optimal threshold and explains "
          "the result in plain English.", kicker="Decisions")
st.warning("**Academic demo only, not a real credit decision.** The model was trained on historical Kaggle data from "
           "an undisclosed country; scores are risk scores (not probabilities) and the costs behind the threshold are "
           "assumptions.", icon=":material/gavel:")


def load(row, label, ex=None):
    st.session_state["base"] = row
    st.session_state["base_label"] = label
    st.session_state["example"] = ex
    for key, value in form_values(row).items():
        st.session_state[f"in_{key}"] = value


if "base" not in st.session_state:
    load(defaults, TYPICAL)

st.subheader("1. Start from an applicant")
cols = st.columns(4)
for col, ex in zip(cols[:3], meta["examples"]):
    col.button(f"Load {ex['label'].lower()} example", on_click=load, args=(ex["row"], f"{ex['label']} example (a real "
               "holdout applicant)", ex), width="stretch",
               icon={"Low risk": ":material/check_circle:", "Medium risk": ":material/remove_circle:",
                     "High risk": ":material/error:"}[ex["label"]])
cols[3].button("Reset to typical applicant", on_click=load, args=(defaults, TYPICAL), width="stretch",
               icon=":material/restart_alt:")
st.caption(f"Starting point: **{st.session_state['base_label']}**. Every model feature not on the form keeps this "
           "starting applicant's value (training medians / most common categories for the typical applicant).")

st.subheader("2. Applicant details")
num_fields = {k: (label, dec, help_) for k, label, dec, help_ in FORM_FIELDS}
SPEC = {"age": (18.0, 80.0, 1.0), "income": (0.0, None, 5_000.0), "credit": (0.0, None, 10_000.0),
        "annuity": (0.0, None, 1_000.0), "goods_price": (0.0, None, 10_000.0), "years_employed": (0.0, 60.0, 0.5),
        "ext_source_mean": (0.0, 1.0, 0.01), "prev_refused": (0.0, 50.0, 1.0), "late_share": (0.0, 1.0, 0.01)}
c = st.columns(3)
for i, (key, (label, dec, help_)) in enumerate(num_fields.items()):
    lo, hi, step = SPEC[key]
    c[i % 3].number_input(label, min_value=lo, max_value=hi, step=step, format=f"%.{dec}f", key=f"in_{key}",
                          help=help_)
c = st.columns(4)
for col, (key, feature, label) in zip(c, FORM_CATEGORIES):
    options = categories[feature]
    if st.session_state.get(f"in_{key}") not in options:
        st.session_state[f"in_{key}"] = defaults[feature]
    col.selectbox(label, options, key=f"in_{key}")

inputs = {k: st.session_state[f"in_{k}"] for k in list(num_fields) + [k for k, _, _ in FORM_CATEGORIES]}
X = build_features(st.session_state["base"], inputs, features, categories)
prob, contrib, base_value = score(L.booster(), X)
reject = prob >= threshold
preds = L.parquet("holdout_predictions.parquet")
percentile = float((preds["p_lgbm"] < prob).mean())

st.subheader("3. Result")
c1, c2 = st.columns([2, 3])
with c1:
    ui.chart(C.gauge(prob, threshold), title="Risk score (0 = safest, 1 = riskiest)",
             subtitle=f"Red zone = rejected (score ≥ {threshold:.2f})")
with c2:
    if reject:
        st.error(f"**Decision: REJECT** — score {prob:.3f} is at or above the cost-optimal threshold {threshold:.2f}.",
                 icon=":material/cancel:")
    else:
        st.success(f"**Decision: APPROVE** — score {prob:.3f} is below the cost-optimal threshold {threshold:.2f}.",
                   icon=":material/check_circle:")
    ui.metrics_row([("Risk score", f"{prob:.3f}", "LightGBM output; not a calibrated probability"),
                    ("Riskier than", S.pct(percentile, 1), "Share of holdout applicants with a lower score"),
                    ("Threshold", f"{threshold:.2f}", "Chosen on training out-of-fold predictions")], columns=3)
    ex = st.session_state.get("example")
    if ex is not None and form_values(st.session_state["base"]) == {k: inputs[k] for k in form_values(st.session_state["base"])}:
        st.caption(f"Unchanged real holdout applicant: stored model score {ex['score']:.3f} (reproduced above); actual "
                   f"outcome: {'defaulted' if ex['actual'] == 1 else 'repaid'}.")

st.markdown("**Top 5 reasons behind this score** (TreeSHAP contributions, log-odds)")
r = reasons(contrib, X.iloc[0], top_n=5)
for _, row in r.iterrows():
    icon = "🔺" if row["effect"] == "raises risk" else "🔻"
    st.markdown(f"{icon} **{row['reason']}** — {row['effect']} ({row['shap_log_odds']:+.3f})")
top10 = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
other = float(contrib.sum() - top10.sum())
labels = list(top10.index) + ["Other features"]
values = list(top10.values) + [other]
ui.chart(C.waterfall(labels, values, base_value), title="How each feature moved the score",
         subtitle="Red bars push towards rejection, blue bars towards approval; the total is the model's log-odds",
         key="pred_wf")
ui.takeaway(f"This applicant scores {prob:.3f}, riskier than {percentile:.1%} of holdout applicants, so the policy would "
            f"**{'reject' if reject else 'approve'}** them. The biggest single factor is "
            f"*{r['reason'].iloc[0].lower()}*, which {r['effect'].iloc[0]}.")

with st.expander("How this works (technical details)", icon=":material/settings:"):
    st.markdown(f"""
- The form edits the raw application values of the starting applicant. Unchanged fields keep their exact values, so a
  loaded example reproduces its real holdout score.
- Engineered features are recomputed with the project's own code: `src.cleaning.clean_application` (employment
  placeholder, age, years employed) and `src.features.add_application_ratios` (credit/income, annuity/income,
  annuity/credit, goods/credit, external-score mean and spread).
- Changing the average external score shifts the applicant's available external scores by the same amount (their
  spread is kept). Changing refused applications updates the previous-application counts and shares; the late-payment
  share is set directly. All other {len(features)} model features keep the starting applicant's values.
- The model is the final LightGBM saved as a text file and scored with LightGBM's `Booster`; reasons come from its
  built-in TreeSHAP (`pred_contrib=True`) and `src.explain.describe_feature`.
""")
