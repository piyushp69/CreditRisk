import streamlit as st

from utils import loaders as L
from utils import style as S
from utils import ui

F = L.facts()
ds, fa, tf, cl = F["dataset"], F["fairness"], F["thin_file"], F["clustering"]
best = L.best_model()
opt, m05 = L.metrics_at("cost-optimal"), L.metrics_at("0.5")
b = opt.loc[best]
groups = L.csv("thin_file_auc.csv").set_index("group")
abl = L.csv("ablation.csv").set_index("variant")
imp = L.csv("shap_importance.csv")
prof = L.csv("cluster_profiles.csv")
fair = L.csv("fairness.csv")
g = fair[(fair["attribute"] == "Gender")].set_index(["model", "group"])
gap_with = g.loc[("with CODE_GENDER", "F"), "approval_rate"] - g.loc[("with CODE_GENDER", "M"), "approval_rate"]
gap_without = g.loc[("without CODE_GENDER", "F"), "approval_rate"] - g.loc[("without CODE_GENDER", "M"), "approval_rate"]
losing_at_05 = int((m05["saving_vs_approve_all"] < 0).sum())

ui.header("Conclusion and future scope", "What the project found, what it cannot say, and what to do next.",
          kicker="Wrap-up")

st.subheader("Answers to the research questions")
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown(f"""
**RQ1 — Can alternative data predict loan default?** Yes. All {len(L.MODEL_ORDER)} models rank applicants well above
chance; the best, **{best}**, reaches holdout AUC **{b['AUC']:.4f}** (Gini {b['Gini']:.4f}), and even without bureau
data and external scores LightGBM keeps AUC {abl.loc['Alternative data only', 'AUC all']:.4f}.
""")
with c2, st.container(border=True):
    st.markdown(f"""
**RQ2 — Does it work without a Credit Bureau history?** Mostly. For the
{int(groups.loc['Bureau history: no', 'applicants']):,} holdout applicants without a bureau record LightGBM's AUC is
**{groups.loc['Bureau history: no', 'LightGBM AUC']:.4f}** ({abl.loc['Alternative data only', 'AUC no-bureau']:.4f}
with alternative data only). But only {tf['neither_holdout']:,} holdout applicants are truly new to credit, too few to
judge.
""")
c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown(f"""
**RQ3 — Which threshold minimises cost?** Each model's own: chosen on training data, {best}'s cut-off of
**{b['threshold']:.2f}** saves **{b['saving_vs_approve_all']:.2%}** of expected cost versus approving everyone
(approval {b['approval_rate']:.2%}). At the conventional 0.5, {losing_at_05} of the {len(L.MODEL_ORDER)} models cost more
than approving everyone. The exact cut-off depends on the assumed costs.
""")
with c2, st.container(border=True):
    st.markdown(f"""
**RQ4 — What drives rejections, and is it fair?** Low external scores (`{imp['feature'].iloc[0]}`), loan structure and
past repayment behaviour, but also gender. Without `CODE_GENDER` the female–male approval gap narrows from
{gap_with * 100:.2f} to {gap_without * 100:.2f} percentage points at a cost of {fa['auc_without'] - fa['auc_with']:+.4f}
AUC; proxies (gender predictable with AUC {fa['proxy_auc']:.4f}) keep part of it.
""")

st.subheader("What the model comparison showed")
st.markdown(f"""
- **{best} is the best model** on every ranking metric and, at its cost-optimal threshold, on precision
  ({b['Precision']:.4f}), recall ({b['Recall']:.4f}) and F1 ({b['F1']:.4f}); a single decision tree is weakest
  (AUC {opt['AUC'].min():.4f}).
- **Accuracy misleads:** approving everyone is {F['models']['approve_all_accuracy']:.2%} accurate; the most "accurate"
  models are those that reject the fewest applicants.
- **Bias–variance:** the linear model and the shallow tree underfit, unrestricted trees overfit, bagging and
  regularised boosting control variance.
- **Clustering** found {cl['chosen_k']} segments with default rates from {prof['default_rate'].min():.2%} to
  {prof['default_rate'].max():.2%}; useful for describing risk groups, not for improving the score
  ({cl['holdout_auc_change']:+.4f} AUC).
""")

st.subheader("Limitations")
st.markdown("""
- The country, currency and exact default definition are undisclosed; loss and margin are **assumed**.
- Scores are not calibrated probabilities (class weighting inflates them).
- Outcomes exist only for loans that were granted (selection bias); no reject inference.
- No application dates, so no out-of-time validation; a single random holdout.
- The monthly behavioural tables (bureau balance, card and point-of-sale histories) were left out; small
  hyperparameter searches; XGBoost untuned.
- Fairness checked only for gender and age; K-Means segments overlap (low silhouette).
""")

st.subheader("Future scope")
st.markdown("""
- Add the excluded monthly tables (bureau balance, card and point-of-sale histories), especially for thin-file
  applicants.
- Calibrate the scores (e.g. isotonic regression on out-of-fold predictions) so they can be read as probabilities.
- Validate costs with real recovery and pricing data; separate thresholds for cash and revolving loans.
- Fairness-aware training or post-processing, and monitoring of approval and wrongful-rejection gaps after deployment.
- External alternative data (with consent) and small, growing starter limits for applicants with no history at all.
""")

st.subheader("References")
st.markdown("""
- Home Credit Group (2018). *Home Credit Default Risk* [Kaggle competition dataset].
  https://www.kaggle.com/competitions/home-credit-default-risk
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees.* Wadsworth.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS 29*.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS 30*.
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification
  algorithms for credit scoring. *European Journal of Operational Research*, 247(1), 124–136.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
- Rousseeuw, P. J. (1987). Silhouettes. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating
  binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.

The full reference list is in `reports/report.md`.
""")

st.subheader("About")
with st.container(border=True):
    st.markdown("""
**Piyush Priyanshu** · Reg. No. **12412878**
Lovely Professional University · **INT234 Predictive Analytics** · Academic Task 2

Project code and notebooks: **[GitHub repository link — add after publishing]**
""")
