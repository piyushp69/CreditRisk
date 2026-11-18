import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds, eda = F["dataset"], F["eda"]
overall = eda["overall_rate_clean"]
FMT = {"applicants": "{:,}", "defaults": "{:,}", "share": "{:.2%}", "default_rate": "{:.2%}"}
LABELS = {"NAME_CONTRACT_TYPE": "Contract type", "CODE_GENDER": "Gender", "FLAG_OWN_CAR": "Owns a car",
          "FLAG_OWN_REALTY": "Owns real estate", "NAME_INCOME_TYPE": "Income type", "NAME_EDUCATION_TYPE": "Education",
          "NAME_FAMILY_STATUS": "Family status", "NAME_HOUSING_TYPE": "Housing", "OCCUPATION_TYPE": "Occupation",
          "ORGANIZATION_TYPE": "Employer type", "NAME_TYPE_SUITE": "Accompanied by", "CNT_CHILDREN": "Number of children",
          "WEEKDAY_APPR_PROCESS_START": "Weekday of application", "REGION_RATING_CLIENT": "Region rating (1 = best)"}


def rates(var):
    return L.eda_rates(var)


def extremes(d, min_share=0.01):
    """Highest and lowest default-rate groups among groups with at least min_share of applicants."""
    big = d[d["share"] >= min_share]
    return big.loc[big["default_rate"].idxmax()], big.loc[big["default_rate"].idxmin()]


ui.header("Exploratory data analysis", f"How the default rate varies across the {ds['n_clean']:,} cleaned applicants "
          f"(overall {overall:.2%}). Every chart is interactive; the dashed line marks the overall rate.",
          kicker="Data")

# ---------------------------------------------------------------- age
st.subheader("Age")
age = rates("AGE_GROUP")
ui.chart(C.rate_bars(age, horizontal=False, overall=overall), title="Default rate by age group (years)",
         data=age[["group", "applicants", "defaults", "share", "default_rate"]], formats=FMT)
ui.takeaway(f"Risk falls steadily with age: {age['default_rate'].iloc[0]:.2%} at {age['group'].iloc[0]} versus "
            f"{age['default_rate'].iloc[-1]:.2%} at {age['group'].iloc[-1]}. Age is a strong signal that needs no "
            "credit history.")

# ---------------------------------------------------------------- categorical explorer
st.subheader("Default rate by applicant category")
var = st.selectbox("Choose a variable", list(LABELS), format_func=lambda v: f"{LABELS[v]} ({v})", index=4)
d = rates(var)
ui.chart(C.rate_bars(d, overall=overall), title=f"Default rate by {LABELS[var].lower()}",
         subtitle="Sorted by default rate" if d["kind"].iloc[0] == "categorical" else "In natural order",
         data=d[["group", "applicants", "defaults", "share", "default_rate"]], formats=FMT)
hi, lo = extremes(d)
ui.takeaway(f"Among groups holding at least 1% of applicants, **{hi['group']}** has the highest default rate "
            f"({hi['default_rate']:.2%}, {hi['applicants']:,} applicants) and **{lo['group']}** the lowest "
            f"({lo['default_rate']:.2%}, {lo['applicants']:,}), a spread of "
            f"{(hi['default_rate'] - lo['default_rate']) * 100:.2f} percentage points. Very small groups are noisy, so "
            "check the applicant counts in the table.")

# ---------------------------------------------------------------- external scores
st.subheader("External credit scores")
hist = L.csv("eda_histograms.csv")
ext = L.csv("eda_ext_sources.csv").set_index("feature")
src = st.segmented_control("Score", ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_MEAN"],
                           default="EXT_SOURCE_MEAN")
src = src or "EXT_SOURCE_MEAN"
ui.chart(C.density_lines(hist, src, f"{src} (0–1)"), title=f"Distribution of {src} for repaid vs defaulted applicants",
         data=hist[hist["variable"] == src].drop(columns="variable"),
         formats={"bin_left": "{:.3f}", "bin_right": "{:.3f}", "density": "{:.3f}", "count": "{:,}"})
st.dataframe(ext.style.format({"pct_missing": "{:.1f}%", "mean_repaid": "{:.4f}", "mean_default": "{:.4f}"}))
dec = rates("EXT_SOURCE_MEAN_DECILE")
ui.takeaway(f"Defaulters have clearly lower external scores (e.g. EXT_SOURCE_3 mean {ext.loc['EXT_SOURCE_3', 'mean_default']:.3f} "
            f"vs {ext.loc['EXT_SOURCE_3', 'mean_repaid']:.3f}). Averaging the three scores gives the sharpest single "
            f"signal in the data: the default rate falls from {dec['default_rate'].iloc[0]:.2%} in the lowest decile "
            f"of EXT_SOURCE_MEAN to {dec['default_rate'].iloc[-1]:.2%} in the highest.")

# ---------------------------------------------------------------- affordability
st.subheader("Affordability")
t1, t2, t3 = st.tabs(["Credit / income", "Annuity / income", "External score (mean) deciles"])
for tab, var_d, name in [(t1, "CREDIT_INCOME_RATIO_DECILE", "credit / income"),
                         (t2, "ANNUITY_INCOME_RATIO_DECILE", "annuity / income"),
                         (t3, "EXT_SOURCE_MEAN_DECILE", "average external score")]:
    with tab:
        d = rates(var_d)
        ui.chart(C.rate_bars(d, overall=overall), title=f"Default rate by decile of {name} (D1 = lowest)",
                 data=d[["group", "applicants", "defaults", "default_rate"]], formats=FMT, key=f"aff_{var_d}")
        spread = d["default_rate"].max() - d["default_rate"].min()
        ui.takeaway(f"Across the {len(d)} deciles of {name}, the default rate varies by {spread * 100:.2f} percentage points "
                    f"(from {d['default_rate'].min():.2%} to {d['default_rate'].max():.2%})."
                    + (" On their own, affordability ratios are weak and non-monotonic signals; they matter mainly in "
                       "combination with other features." if "EXT" not in var_d else
                       " This is by far the steepest gradient of any single feature."))

# ---------------------------------------------------------------- correlations
st.subheader("Which numeric features move with default?")
corr = L.csv("eda_correlations.csv")
t1, t2 = st.tabs(["Application columns", "Engineered features"])
for tab, s in [(t1, "application columns"), (t2, "engineered features")]:
    with tab:
        d = corr[corr["set"] == s]
        ui.chart(C.corr_bars(d), title=f"Top 15 {s} by |correlation| with TARGET",
                 subtitle="Orange = higher value, more default; blue = higher value, less default",
                 data=d, formats={"corr": "{:+.4f}"}, key=f"corr_{s}")
        top = d.iloc[0]
        ui.takeaway(f"The strongest linear signal is **{top['feature']}** (r = {top['corr']:+.4f}), and even that is "
                    "a weak correlation: default cannot be predicted from one variable, so a multivariate, non-linear "
                    "model is needed.")

# ---------------------------------------------------------------- thin file
st.subheader("Thin-file view: who has a credit history?")
c1, c2 = st.columns(2)
bureau, prev = rates("HAS_BUREAU_HISTORY"), rates("HAS_PREV_APPLICATION")
with c1:
    ui.chart(C.rate_bars(bureau, horizontal=False, overall=overall, height=300), title="Credit Bureau record?",
             data=bureau[["group", "applicants", "defaults", "default_rate"]], formats=FMT)
with c2:
    ui.chart(C.rate_bars(prev, horizontal=False, overall=overall, height=300), title="Previous Home Credit application?",
             data=prev[["group", "applicants", "defaults", "default_rate"]], formats=FMT)
combo = rates("HISTORY_GROUP")
ui.chart(C.rate_bars(combo, overall=overall), title="Combined history groups",
         data=combo[["group", "applicants", "defaults", "default_rate"]], formats=FMT)
nb = bureau.set_index("group")
ui.takeaway(f"{eda['no_bureau']:,} applicants ({nb.loc['No', 'share']:.2%}) have no Credit Bureau record and default "
            f"more often ({nb.loc['No', 'default_rate']:.2%} vs {nb.loc['Yes', 'default_rate']:.2%}). But "
            f"{eda['no_bureau_with_prev_share']:.2%} of them have a previous Home Credit application, so the lender's "
            f"own history is available as alternative data. Only {eda['neither']:,} applicants have neither.")

# ---------------------------------------------------------------- behaviour
st.subheader("Past behaviour with Home Credit")
t1, t2 = st.tabs(["Late instalments", "Refused applications"])
for tab, var_b, name in [(t1, "INST_LATE_SHARE_BAND", "share of previous instalments paid late"),
                         (t2, "PREV_REFUSED_SHARE_BAND", "share of previous applications refused")]:
    with tab:
        d = rates(var_b)
        ui.chart(C.rate_bars(d, horizontal=False, overall=overall), title=f"Default rate by {name}",
                 data=d[["group", "applicants", "defaults", "default_rate"]], formats=FMT, key=f"beh_{var_b}")
        known = d[~d["group"].str.startswith("No ")]
        ui.takeaway(f"Behaviour is a strong alternative-data signal: the default rate rises from "
                    f"{known['default_rate'].iloc[0]:.2%} ({known['group'].iloc[0]}) to "
                    f"{known['default_rate'].iloc[-1]:.2%} ({known['group'].iloc[-1]}).")

# ---------------------------------------------------------------- numeric explorer
st.subheader("Numeric explorer")
num_vars = ["AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
            "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "GOODS_CREDIT_RATIO", "INST_LATE_SHARE", "PREV_REFUSED_SHARE",
            "BUR_ACTIVE_SHARE", "DAYS_LAST_PHONE_CHANGE"]
nv = st.selectbox("Numeric feature", num_vars)
ui.chart(C.density_lines(hist, nv, nv), title=f"Distribution of {nv} for repaid vs defaulted applicants",
         subtitle="Middle 99% of values; the shapes are compared per class, so the rare defaults are visible",
         data=hist[hist["variable"] == nv].drop(columns="variable"),
         formats={"bin_left": "{:,.3f}", "bin_right": "{:,.3f}", "density": "{:.4g}", "count": "{:,}"})
ui.takeaway("Where the two curves separate, the feature helps tell defaulters from good applicants; where they "
            "overlap, it carries little information on its own.")

# ---------------------------------------------------------------- key findings
st.subheader("Key findings")
edu = rates("NAME_EDUCATION_TYPE").set_index("group")
late = rates("INST_LATE_SHARE_BAND")
st.markdown(f"""
1. **Strong class imbalance:** {ds['default_rate']:.2%} defaults; accuracy is a misleading metric here.
2. **External scores are the strongest signal:** default rate {dec['default_rate'].iloc[0]:.2%} in the lowest
   EXT_SOURCE_MEAN decile vs {dec['default_rate'].iloc[-1]:.2%} in the highest.
3. **Age matters:** {age['default_rate'].iloc[0]:.2%} at {age['group'].iloc[0]} vs {age['default_rate'].iloc[-1]:.2%}
   at {age['group'].iloc[-1]}.
4. **Education matters:** lower secondary {edu.loc['Lower secondary', 'default_rate']:.2%} vs higher education
   {edu.loc['Higher education', 'default_rate']:.2%}.
5. **Past behaviour is alternative data:** never-late payers default at {late['default_rate'].iloc[0]:.2%}, those late
   on more than half of their instalments at {late[late['group'] == '>50%']['default_rate'].iloc[0]:.2%}.
6. **Thin-file applicants are riskier** ({nb.loc['No', 'default_rate']:.2%} without a bureau record) but most of them
   have Home Credit history.
7. **The DAYS_EMPLOYED placeholder is informative** ({ds['anomaly_rate']:.2%} vs {ds['valid_rate']:.2%}), so it is
   kept as a flag.
""")
