import streamlit as st

from utils import charts as C
from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds, cfg = F["dataset"], F["config"]
best = L.best_model()
opt, m05 = L.metrics_at("cost-optimal"), L.metrics_at("0.5")
b = opt.loc[best]
thin = L.csv("thin_file_auc.csv").set_index("group")

ui.header(
    "Predicting loan default for thin-file borrowers",
    f"Many people cannot get a loan because they have **little or no credit history** (a *thin file*): credit "
    f"bureaus know too little about them to score them, so lenders reject them or charge them more. This project "
    f"uses real lending data from **Home Credit Group** ({ds['n_raw']:,} loan applications) to test whether "
    f"*alternative data* (the application form, earlier applications and repayment behaviour) can predict who will "
    f"default, which of {len(L.MODEL_ORDER)} machine-learning models does it best, and how a lender should turn the "
    f"scores into "
    f"fair, explainable decisions.",
    kicker="INT234 Predictive Analytics · Academic Task 2")

st.subheader("Headline results")
ui.metrics_row([
    ("Best model", best, f"Highest holdout AUC of the {len(L.MODEL_ORDER)} models; it also wins every other ranking "
                         "metric."),
    (f"{best} AUC", S.f4(b["AUC"]),
     "Probability that a random defaulter is scored riskier than a random good applicant (holdout, evaluated once)."),
    (f"{best} F1", S.f4(b["F1"]),
     f"Balance of precision and recall for defaulters at its cost-optimal threshold ({b['threshold']:.2f})."),
    ("Cost saving", S.pct(b["saving_vs_approve_all"]),
     "Expected cost saved compared with approving every applicant, under the assumed loss and margin."),
    ("Approval rate", S.pct(b["approval_rate"]), "Share of holdout applicants approved at the cost-optimal threshold."),
])
ui.chart(C.model_bars(opt["AUC"], ytitle="Holdout AUC", height=320,
                      yrange=[0.5, float(opt["AUC"].max()) + 0.05], best=best),
         title=f"Ranking quality of the {len(L.MODEL_ORDER)} models (holdout AUC, higher is better)",
         data=opt[["AUC", "Gini", "PR_AUC", "KS"]].reset_index(),
         formats={c: "{:.4f}" for c in ["AUC", "Gini", "PR_AUC", "KS"]})
worst = opt["AUC"].idxmin()
ui.takeaway(f"All {len(L.MODEL_ORDER)} models rank applicants far better than guessing (AUC 0.5). **{best}** is best "
            f"(AUC {b['AUC']:.4f}) and **{worst}** weakest ({opt.loc[worst, 'AUC']:.4f}). At its cost-optimal "
            f"threshold {best} approves {b['approval_rate']:.2%} of applicants and cuts expected cost by "
            f"{b['saving_vs_approve_all']:.2%} compared with approving everyone; for applicants with no Credit "
            f"Bureau record its AUC is still {thin.loc['Bureau history: no', 'LightGBM AUC']:.4f}.")

st.subheader("Research questions")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**RQ1.** Can alternative data predict loan default?\n\n"
                "**RQ2.** How well do the models work for applicants with no Credit Bureau history?")
with c2:
    st.markdown("**RQ3.** Which decision threshold minimises the lender's expected cost?\n\n"
                "**RQ4.** What drives rejections, and are decisions fair across gender and age?")

st.subheader("How the project was built")
files = L.csv("files.csv")
n_tables = int((files["status"] == "used").sum())
ui.diagram(ui.flow_dot([f"Data\\n(Kaggle, {n_tables} files)", "Cleaning", "EDA", "Feature\\nengineering",
                        f"Clustering\\n(K-Means, k={F['clustering']['chosen_k']})",
                        f"{len(L.MODEL_ORDER)} models\\n(tuned with CV)", "Comparison\\n(holdout, once)",
                        "Decision\\n(cost threshold)"], highlight=[5, 6]))
st.caption(f"The data are split once into {ds['n_train']:,} training and {ds['n_holdout']:,} holdout applicants "
           f"(stratified, random_state={cfg['random_state']}). Everything is fitted on the training part; the "
           f"holdout is used once per model to report results.")

st.subheader("How to use this website")
st.markdown(f"""
| Page | What you will find |
|---|---|
| **Dataset** | Where the data comes from, which files were used, the class imbalance, a searchable column dictionary |
| **Preprocessing** | Missing values, the employment anomaly, encoding, scaling, the train/holdout split, every engineered feature |
| **Exploratory analysis** | Interactive default-rate charts for any variable, external scores, affordability, thin-file groups |
| **Clustering** | {F['clustering']['chosen_k']} K-Means applicant segments, how *k* was chosen, and whether segments help the model |
| **How each model works** | One tab per model: plain-English explanation, diagram, settings, results, strengths and weaknesses |
| **Model comparison** | The main page: every metric for every model, a threshold slider, overfitting check and the verdict |
| **Business decision** | Change the cost assumptions and watch the best threshold and the saving move |
| **Thin-file & explainability** | Results for applicants without bureau history, and SHAP reasons behind decisions |
| **Fairness** | Approval and error rates by gender and age, with and without gender as an input |
| **Live risk predictor** | Enter an applicant and get a score, a decision and the top reasons |
| **Conclusion** | Answers to the research questions, limitations, future work, references |
""")
st.caption("Open the menu (» on phones) to switch pages. Charts are interactive: hover for values, drag to zoom, "
           "click legend entries to hide or show a model; every chart has a table view underneath.")
