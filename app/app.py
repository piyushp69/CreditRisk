"""Project website: Predicting loan default for thin-file borrowers (Home Credit Default Risk).

Run from the project root (after `python app/build_assets.py` has created app/assets/ and app/models/):

    python -m streamlit run app/app.py
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
for path in (str(APP_DIR.parent), str(APP_DIR)):  # project root (for src/) and app/ (for utils/)
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Loan Default Risk · Home Credit", page_icon=":material/credit_score:", layout="wide",
                   initial_sidebar_state="auto",
                   menu_items={"About": "INT234 Predictive Analytics project (Academic Task 2) by Piyush Priyanshu, "
                                        "Lovely Professional University. Data: Home Credit Default Risk (Kaggle)."})

from utils import loaders, ui  # noqa: E402

ui.inject_css()
st.logo(str(APP_DIR / "assets" / "logo.svg"), size="large")

missing = loaders.missing_assets()
if missing:
    st.title("Loan Default Risk")
    st.error("The website's data bundle has not been built yet (missing: " + ", ".join(missing) + ").")
    st.code("python app/build_assets.py", language="bash")
    st.stop()

PAGES = APP_DIR / "site_pages"  # not "pages/": that name triggers Streamlit's legacy auto-navigation for deep links
pages = {
    "Overview": [st.Page(PAGES / "01_home.py", title="Home", icon=":material/home:", default=True)],
    "Data": [
        st.Page(PAGES / "02_dataset.py", title="Dataset", icon=":material/database:"),
        st.Page(PAGES / "03_preprocessing.py", title="Preprocessing", icon=":material/cleaning_services:"),
        st.Page(PAGES / "04_eda.py", title="Exploratory analysis", icon=":material/query_stats:"),
        st.Page(PAGES / "05_clustering.py", title="Clustering", icon=":material/bubble_chart:"),
    ],
    "Models": [
        st.Page(PAGES / "06_models.py", title="How each model works", icon=":material/account_tree:"),
        st.Page(PAGES / "07_comparison.py", title="Model comparison", icon=":material/leaderboard:"),
    ],
    "Decisions": [
        st.Page(PAGES / "08_business.py", title="Business decision", icon=":material/payments:"),
        st.Page(PAGES / "09_thin_file.py", title="Thin-file & explainability", icon=":material/person_search:"),
        st.Page(PAGES / "10_fairness.py", title="Fairness", icon=":material/balance:"),
        st.Page(PAGES / "11_predictor.py", title="Live risk predictor", icon=":material/calculate:"),
    ],
    "Wrap-up": [st.Page(PAGES / "12_conclusion.py", title="Conclusion & future scope", icon=":material/flag:")],
}
page = st.navigation(pages, position="sidebar", expanded=True)
with st.sidebar:
    st.caption("Data: Home Credit Default Risk, official Kaggle competition (Home Credit Group). Amounts are in the "
               "data's undisclosed currency. Academic project, not a lending tool.")
page.run()
