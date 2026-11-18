import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds, cfg, mods = F["dataset"], F["config"], F["models"]
n_in = mods["n_inputs"]
max_auc = L.metrics_at("0.5")["AUC"].max()

ui.header("Preprocessing", "How raw application data was cleaned and turned into model inputs without leaking the "
          "holdout.", kicker="Data")

# ---------------------------------------------------------------- missing values
st.subheader("1. Missing values")
miss = L.csv("missing_values.csv")
ui.metrics_row([
    ("Columns with gaps", f"{ds['missing_columns']} of {ds['n_raw_columns']}", "Columns with at least one missing value"),
    ("More than 50% missing", S.num(ds["missing_over_50"]), "Columns missing for more than half of the applicants"),
    ("…of which building info", S.num(ds["building_over_50"]), "*_AVG, *_MODE, *_MEDI columns about the home"),
    ("Worst column", f"{ds['max_pct_missing']:.1f}%", "Highest share of missing values in one column"),
])
top = miss.head(30).iloc[::-1]
fig = go.Figure(go.Bar(x=top["pct_missing"] / 100, y=top["column"], orientation="h",
                       marker_color=[S.series(0) if b else S.series(1) for b in top["is_building"]],
                       marker_cornerradius=3, customdata=top["n_missing"],
                       hovertemplate="<b>%{y}</b><br>%{x:.1%} missing (%{customdata:,} applicants)<extra></extra>"))
fig.add_vline(x=0.5, line=dict(color=S.muted(), width=1, dash="dash"), annotation_text="50%")
C.layout(fig, xtitle="Share of applicants missing", height=720, legend=False, xpct=True, xrange=[0, 1])
ui.chart(fig, title="Top 30 columns by share missing",
         subtitle="Blue = building-information columns, orange = other columns",
         data=miss, formats={"pct_missing": "{:.2f}", "n_missing": "{:,}"})
ui.takeaway(f"{ds['building_over_50']} of the {ds['missing_over_50']} columns that are more than half empty describe "
            "the applicant's building. Rows were **not** deleted: missingness itself is informative (for example, "
            "no external score usually means no credit record), so each model handles gaps in its own way.")
ui.table(
    pd.DataFrame({
        "Model": L.MODEL_ORDER,
        "Missing numeric values": ["Median imputation + one 0/1 indicator per column", "Median imputation",
                                   "Median imputation", "Native (learns a default branch)",
                                   "Native (learns a default branch)"],
        "Missing categories": ["Own 'MISSING' level"] * 3 + ["Native", "Native"],
        "Fitted on": ["Training folds only (inside the sklearn Pipeline)"] * 3 + ["Not needed"] * 2}))

# ---------------------------------------------------------------- anomaly
st.subheader("2. The DAYS_EMPLOYED anomaly")
anom = L.csv("days_employed_anomaly.csv")
c1, c2 = st.columns([1, 1])
with c1:
    st.markdown(f"""
`DAYS_EMPLOYED = 365243` (about 1,000 years) is a placeholder, not a real employment length. It affects
**{ds['anomaly_count']:,} applicants ({ds['anomaly_share']:.2%})**: {ds['anomaly_pensioners']:,} pensioners and
{ds['anomaly_unemployed']:,} unemployed.

**Fix:** the value is set to missing and a flag `DAYS_EMPLOYED_ANOM = 1` keeps the information, because these
applicants default at a different rate. Also: the {ds['xna_rows']} rows with `CODE_GENDER = 'XNA'` were dropped,
leaving {ds['n_clean']:,} applicants; extreme incomes (median {ds['income_median']:,.0f}, 99.9th percentile
{ds['income_p999']:,.0f}, maximum {ds['income_max']:,.0f}) were kept because trees are insensitive to outliers.
""")
with c2:
    d = anom.rename(columns={"group": "group"}).assign(defaults=(anom["applicants"] * anom["default_rate"]).round())
    ui.chart(C.rate_bars(d, horizontal=False, overall=ds["default_rate"], height=300),
             title="Default rate: placeholder vs real employment length", data=anom,
             formats={"applicants": "{:,}", "share": "{:.2%}", "default_rate": "{:.2%}"})
ui.takeaway(f"Applicants with the placeholder default at {ds['anomaly_rate']:.2%} against {ds['valid_rate']:.2%} for "
            "the rest, so silently replacing the value would throw away a real signal; the flag keeps it.")

# ---------------------------------------------------------------- encoding and scaling
st.subheader("3. Categorical variables")
st.markdown(f"""
The model table has **{ds['n_categorical']} categorical features** (e.g. income type, occupation, employer type).
- **LightGBM and XGBoost** split on categories directly (native categorical support).
- **Logistic regression, decision tree and random forest** need numbers, so their pipelines use
  `OneHotEncoder(handle_unknown='ignore', min_frequency=0.01)`: one 0/1 column per category; categories seen in less
  than 1% of training applicants are pooled into one "infrequent" column, and categories never seen in training are
  ignored. After preprocessing the logistic regression sees **{n_in['Logistic regression']} inputs** and the tree
  models **{n_in['Decision tree']} inputs** (from {ds['n_features']} features).
""")
st.subheader("4. Feature scaling")
ui.table(pd.DataFrame({
    "Model": L.MODEL_ORDER,
    "Scaling": ["StandardScaler (mean 0, standard deviation 1)", "None", "None", "None", "None"],
    "Why": ["Weights, the L2 penalty and the optimiser are all scale-sensitive: an income in the hundreds of thousands "
            "would otherwise swamp a 0/1 flag",
            "A split compares one feature with a cut-off, so rescaling changes nothing",
            "Same as the decision tree", "Same (trees)", "Same (trees)"]}))
st.caption("K-Means clustering (Clustering page) also standardises its features, because it measures Euclidean "
           "distances.")

# ---------------------------------------------------------------- split and leakage
st.subheader("5. Train / holdout split and leakage prevention")
ui.metrics_row([
    ("Training applicants", S.num(ds["n_train"]), "Used for all fitting, tuning, early stopping and threshold choice"),
    ("Training default rate", S.pct(ds["train_default_rate"], 4), "Stratified split keeps the rate equal"),
    ("Holdout applicants", S.num(ds["n_holdout"]), f"{cfg['holdout_share']:.0%} of the labelled applicants, used once per model"),
    ("Holdout default rate", S.pct(ds["holdout_default_rate"], 4), "Stratified split keeps the rate equal"),
])
ui.diagram(
    'digraph { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", color="#c3c2b7", '
    'fillcolor="#f2f1ec", fontname="Helvetica", fontsize=11]; edge [color="#898781", fontname="Helvetica", fontsize=10]; '
    f'all [label="{ds["n_clean"]:,} labelled\\napplicants"]; '
    f'tr [label="Training set\\n{ds["n_train"]:,}", fillcolor="#2a78d6", fontcolor="white"]; '
    f'ho [label="Holdout set\\n{ds["n_holdout"]:,}", fillcolor="#eb6834", fontcolor="white"]; '
    f'cv [label="{cfg["cv_folds"]}-fold stratified CV\\n(tuning, model choice)"]; '
    'fit [label="Pipelines fitted on train only\\n(imputation, scaling, encoding,\\nearly stopping, K-Means)"]; '
    'thr [label="Cost-optimal threshold\\nfrom out-of-fold predictions"]; '
    'ev [label="Evaluated ONCE\\nper model"]; '
    f'all -> tr [label="stratified split, random_state={cfg["random_state"]}"]; all -> ho; '
    'tr -> cv; tr -> fit; cv -> thr; fit -> ev; thr -> ev; ho -> ev; }')
st.markdown(f"""
- The split is drawn **once** and saved (`data/processed/split_ids.parquet`); every notebook reloads it.
- Every transformer is inside an sklearn `Pipeline`, so cross-validation refits it on each training fold.
- Boosting models stop adding trees using a {cfg['es_fraction']:.0%} carve-out of the data they are trained on, never
  the fold being scored or the holdout.
- Side-table features summarise only each applicant's own records, so they carry nothing across applicants.
- A guard stops the analysis if any holdout AUC exceeds 0.85 (it never did: the highest holdout AUC is
  {max_auc:.4f}).
""")

# ---------------------------------------------------------------- feature engineering
st.subheader("6. Feature engineering")
fe = L.csv("feature_dictionary.csv")
st.markdown(f"Starting from the cleaned application ({ds['n_raw_columns']} columns), **{len(fe)} features** were "
            f"engineered from {fe['source'].nunique()} tables, giving **{ds['n_features']} model features** per "
            "applicant. Side tables "
            "were first summarised per applicant, then joined.")
group = st.segmented_control("Feature group", ["All"] + list(fe["group"].unique()), default="All")
view = fe if group in (None, "All") else fe[fe["group"] == group]
ui.table(view.rename(columns={"feature": "Feature", "group": "Group", "formula": "Formula", "meaning": "Meaning",
                              "source": "Source table"}), height=420)
