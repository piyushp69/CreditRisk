import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds = F["dataset"]

ui.header("Dataset", "Where the data comes from, what was used, and why defaults are hard to find.", kicker="Data")

st.subheader("Source and authenticity")
c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f"""
- **Official Kaggle competition:** *Home Credit Default Risk* — [{ds['kaggle_url']}]({ds['kaggle_url']})
- **Data provider:** Home Credit Group, a consumer lender that focuses on people with little or no credit history.
- **How it was obtained:** downloaded with the Kaggle API after accepting the competition rules; raw files are not
  redistributed (this website only shows aggregates and model outputs).
- **Task:** predict `TARGET = 1`, a *client with payment difficulties* (late by more than X days on at least one of
  the first Y instalments; X and Y are not disclosed by the provider).
""")
with c2:
    ui.metrics_row([("Labelled applications", S.num(ds["n_raw"]), "Rows of application_train"),
                    ("Columns", S.num(ds["n_raw_columns"]), "Columns of application_train")], columns=2)
    ui.metrics_row([("Defaults", S.num(ds["n_defaults"]), "Applications with TARGET = 1"),
                    ("Default rate", S.pct(ds["default_rate"]), "Share of applications with TARGET = 1")], columns=2)

st.subheader("Files used and excluded")
files = L.csv("files.csv")
ui.table(files.rename(columns={"file": "File", "status": "Status", "rows": "Rows", "columns": "Columns",
                               "contents": "Contents / reason"}),
         formats={"Rows": "{:,.0f}", "Columns": "{:,.0f}"})
st.caption(f"`application_test` ({ds['n_test_unlabelled']:,} rows) has no labels, so it cannot be used to measure "
           "accuracy. Excluded tables are large monthly histories left out of the project's scope.")

st.subheader("Class imbalance: defaults are rare")
counts = [ds["n_repaid"], ds["n_defaults"]]
fig = go.Figure(go.Bar(x=["Repaid (TARGET = 0)", "Default (TARGET = 1)"], y=counts,
                       marker_color=[S.series(0), S.series(1)], marker_cornerradius=4,
                       text=[f"{n:,} ({n / sum(counts):.2%})" for n in counts], textposition="outside",
                       cliponaxis=False, hovertemplate="%{x}<br>%{y:,} applications<extra></extra>"))
C.layout(fig, ytitle="Applications", height=340, legend=False, yrange=[0, max(counts) * 1.18])
ui.chart(fig, title="TARGET distribution in application_train")
ui.takeaway(f"Only {ds['default_rate']:.2%} of applicants default: {ds['repaid_per_default']:.2f} repaid loans for "
            f"every default. A model that simply approves everyone would be {1 - ds['default_rate']:.2%} *accurate* "
            "while catching no defaulter, so this project judges models by AUC, PR-AUC, recall and F1 instead of "
            "accuracy, and gives defaults extra weight during training.")

st.subheader("Searchable column dictionary")
desc = L.csv("column_dictionary.csv")
c1, c2 = st.columns([2, 1])
query = c1.text_input("Search column names and descriptions", placeholder="e.g. EXT_SOURCE, income, overdue")
table_choice = c2.selectbox("Table", ["All tables"] + sorted(desc["Table"].unique()))
view = desc if table_choice == "All tables" else desc[desc["Table"] == table_choice]
if query:
    q = query.lower()
    view = view[view["column"].str.lower().str.contains(q, regex=False) |
                view["Description"].str.lower().str.contains(q, regex=False)]
st.caption(f"{len(view):,} of {len(desc):,} columns shown (dictionary supplied by Home Credit with the data).")
ui.table(view, height=380)

st.subheader("Data limitations")
st.markdown("""
- **Undisclosed context:** the country, the currency and the exact definition of default are not published, so cost
  results are in "currency units" and cannot be compared with a specific market.
- **Only granted loans have outcomes:** applicants who were rejected in the past are not in the labelled data
  (selection bias); no reject inference was attempted.
- **No application dates:** an out-of-time validation (train on older loans, test on newer ones) is not possible.
- **Behavioural tables left out:** monthly bureau, card and point-of-sale histories may carry extra signal,
  especially for thin-file applicants.
""")
