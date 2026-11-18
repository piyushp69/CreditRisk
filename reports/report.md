# Predicting Loan Default for Thin-File Borrowers Using Alternative Data

<br>

| | |
|---|---|
| **Name** | Piyush Priyanshu |
| **Reg. No.** | 12412878 |
| **Course** | INT234 Predictive Analytics |
| **Assessment** | Academic Task 2 |
| **University** | Lovely Professional University |
| **Dataset** | Home Credit Default Risk — official Kaggle competition data provided by Home Credit Group |
| **Submission date** | 17 November 2026 |

*Every project number in this report comes from the executed notebooks (`notebooks/01`–`05`) or the files they
save in `reports/`; external statistics are cited. Figures are in `reports/figures/` and are referenced by file name.
The same results are presented interactively in the project website (`app/`, see `README.md`).*

<div style="page-break-after: always;"></div>

## Abstract

Applicants with little or no formal credit history ("thin-file" borrowers) are hard to assess with traditional
bureau-based scoring. Using the official Home Credit Default Risk data (307,507 labelled applicants, 8.07% default
rate), we build a modelling table of 161 features from the loan application, Credit Bureau records, previous Home
Credit applications and instalment repayment history. Five classifiers are compared under one protocol (tuning and
5-fold cross-validation on the training set, one evaluation on a stratified 20% holdout): logistic regression, a
decision tree, a random forest, XGBoost and LightGBM. LightGBM is best on every ranking metric (holdout AUC 0.7819,
Gini 0.5638, PR-AUC 0.2818, KS 0.4237) and, at its cost-optimal threshold, on precision, recall and F1, while the
decision tree is weakest (AUC 0.7269). Accuracy is shown to be misleading: approving everyone is 91.93% accurate, and
the most "accurate" models are those that reject the fewest applicants. Under assumed costs (loss given default 0.60,
lost margin 0.15 of the credit amount), each model's threshold is chosen on out-of-fold training predictions; LightGBM's
threshold of 0.70 cuts holdout cost by 14.04% relative to approving everyone, whereas four of the five models lose
money at the conventional 0.5. K-Means clustering finds six applicant segments whose default rates range from 3.81%
to 13.27%, with past repayment behaviour and refusals marking the riskiest groups. For the 8,755 holdout applicants
without Credit Bureau history LightGBM reaches AUC 0.7645, and a model without bureau data and external scores still
reaches 0.7507 on this group. SHAP analysis shows that external scores, loan structure and repayment behaviour drive
predictions, but also that gender carries weight, a fairness concern that removing the attribute only partly
resolves.

## 1 Introduction

Access to formal credit depends heavily on having a credit history. People who are young, new to a country, or who
have mostly used cash are often "credit invisible" or thin-file: credit bureaus have too little information on them
for a conventional score, so lenders either reject them or price them conservatively. This credit-history gap is a
barrier to financial inclusion. According to the World Bank's Global Findex Database 2025, 79% of adults worldwide
had an account at a bank, another regulated institution or a mobile-money provider in 2024, which still leaves about
1.3 billion adults without one (Klapper et al., 2025). Even in a mature credit market such as the United States, the
Consumer Financial Protection Bureau estimates that 2.7% of adults (about 7.0 million people) had no record at a
nationwide credit bureau in December 2020, and a further 9.8% had a record too thin or too stale to be scored
(Kambara & Luce, 2025); this revises the widely cited 2015 estimate of 26 million "credit invisible" Americans
(Brevoort et al., 2015). Being unbanked and being credit-invisible are not the same thing, but both describe people
whom a conventional bureau score cannot assess.

One proposed remedy is *alternative data*: information beyond the bureau file, such as the application itself
(employment, housing, family, loan terms), the lender's own history with the customer, or payment behaviour on other
obligations. Machine-learning models can combine many such weak signals, but they raise questions of how well they
work for the people who need them most, how lenders should turn scores into decisions, and whether the resulting
decisions are explainable and fair.

This project studies these questions on a large, real consumer-lending dataset published by Home Credit Group, a
lender focused on customers with little or no credit history. It follows the full predictive-analytics workflow
required by the course: data selection and cleaning, exploratory analysis, feature engineering, K-Means
clustering, a comparison of five supervised models (logistic regression, decision tree, random forest, XGBoost and
LightGBM) with tuning, a justified choice of evaluation metrics, a bias–variance discussion, and a decision analysis
covering cost, explainability and fairness.

## 2 Dataset selection and problem definition

### 2.1 Why this dataset

- **Real and authentic:** the official Kaggle competition "Home Credit Default Risk",
  https://www.kaggle.com/competitions/home-credit-default-risk, published by the lender Home Credit Group and
  downloaded with the Kaggle API on 2026-10-27 after accepting the competition rules.
- **Directly relevant to the research topic:** Home Credit lends mainly to people with little or no credit history,
  and the data contain both bureau records and "alternative" information (application form, previous applications,
  repayment behaviour).
- **Large enough for a sound methodology:** 307,511 labelled applications allow a stratified 20% holdout of 61,502
  applicants that is used only once, plus 5-fold cross-validation on the remaining 246,005.
- **Limitations accepted:** the provider does **not disclose the country**; monetary amounts are in an unspecified
  local currency; the exact target definition (X days late on one of the first Y instalments) is not disclosed.

### 2.2 Files used and excluded

- **Used:** `application_train.csv` (307,511 × 122, labelled), `bureau.csv` (1,716,428 × 17),
  `previous_application.csv` (1,670,214 × 37), `installments_payments.csv` (13,605,401 × 8) and
  `HomeCredit_columns_description.csv` (column dictionary). `application_test.csv` (48,744 × 121) was downloaded but
  has no labels, so it is **not** used for evaluation.
- **Excluded:** `bureau_balance.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv` (monthly behavioural
  histories, left out of the initial scope for size and simplicity) and `sample_submission.csv` (Kaggle submission
  format only). See `data/README.md`.

### 2.3 Problem definition and research questions

The task is **binary classification**: predict whether an applicant will have payment difficulties (`TARGET = 1`,
"client with payment difficulties: he/she had late payment more than X days on at least one of the first Y
installments of the loan") from information available at application time. The positive class is rare (8.07%), so
the problem is also an **imbalanced-classification** and **cost-sensitive decision** problem. Four research
questions guide the analysis:

- **RQ1:** Can alternative data predict loan default?
- **RQ2:** How well does the model work for applicants with no Credit Bureau history?
- **RQ3:** Which decision threshold minimises the lender's expected cost?
- **RQ4:** What drives rejections, and are the resulting decisions fair across gender and age groups?

## 3 Data preprocessing

Cleaning is implemented in `src/cleaning.py::clean_application` (notebook 01); all model-specific preprocessing is
inside sklearn pipelines in `src/modeling.py` (notebook 03).

### 3.1 Missing values

- 67 of 122 application columns contain missing values and 41 are more than 50% missing, 39 of them
  building-information columns (`*_AVG`, `*_MODE`, `*_MEDI`, up to 69.9% missing). `EXT_SOURCE_1` is 56.4% missing
  and `EXT_SOURCE_3` 19.8% (`02_missing_values_top30.png`).
- **No rows are deleted for missingness.** Missing values are handled per model: LightGBM and XGBoost learn a
  default split direction for missing values natively; logistic regression uses median imputation plus one
  missing-value indicator column per affected feature (missingness itself can carry signal); the decision tree and
  random forest use median imputation; missing categorical values become their own `MISSING` level. Every imputer is fitted inside
  the pipeline, i.e. on training folds only.
- **Anomaly:** `DAYS_EMPLOYED = 365243` (about 1,000 years) is a placeholder affecting 55,374 applicants (18.01%), of
  whom 55,352 are pensioners and 22 unemployed. Because their default rate differs (5.40% vs 8.66%), the value is set
  to missing and a flag `DAYS_EMPLOYED_ANOM` is kept.
- 4 rows with `CODE_GENDER = 'XNA'` (no defaults) are dropped, leaving 307,507 applicants.
- `AMT_INCOME_TOTAL` has extreme values (median 147,150; 99.9th percentile 900,000; maximum 117,000,000). The 278
  applicants above the 99.9th percentile are reported but kept: tree models are insensitive to outliers, and in ratio
  features an extreme income only affects its own row.
- Side-table aggregates: count features are 0 for applicants without rows in that table; other aggregates stay
  missing (an applicant without bureau history has no "bureau debt ratio").

### 3.2 Categorical variables

The model table has 16 categorical features (e.g. `NAME_INCOME_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`).
LightGBM and XGBoost use them natively (category splits). Logistic regression, the decision tree and the random
forest need numeric input, so their pipelines apply `OneHotEncoder(handle_unknown='ignore', min_frequency=0.01)`: levels seen in fewer
than 1% of training applicants are pooled into one "infrequent" column, which limits dimensionality and protects
against unseen levels at prediction time.

### 3.3 Feature scaling

`StandardScaler` (fitted on training data only) is applied only where the algorithm needs it: logistic regression
(the coefficients, the L2 penalty and the lbfgs optimiser are scale-sensitive) and K-Means clustering (which uses
Euclidean distance, so a feature measured in large units would dominate). Tree-based models (decision tree, random
forest, XGBoost, LightGBM) split on thresholds of one feature at a time, so monotone rescaling does not change them
and no scaling is applied.

### 3.4 Data splitting and leakage prevention

- A stratified 80/20 split by applicant (`random_state = 42`) was drawn once and saved
  (`data/processed/split_ids.parquet`): 246,005 training applicants (8.0730% default) and 61,502 holdout applicants
  (8.0729%).
- All tuning, cross-validation (5-fold stratified), early stopping and threshold selection use the training set
  only. The holdout is evaluated **once per model**.
- All imputation, scaling, encoding and clustering transformers are fitted inside pipelines or on training data
  only; gradient-boosting early stopping uses a 10% carve-out of the data being fitted, never the CV fold being scored.
- Side-table features summarise only each applicant's own records, so building them before the split does not leak
  information between applicants.
- A guard stops the analysis if any holdout AUC exceeds 0.85.

### 3.5 Feature engineering

All features are built in `src/features.py` (notebook 02). Side tables are aggregated per applicant and left-joined,
giving one row per applicant: 307,507 rows and 161 features.

- **Application ratios:** credit/income, annuity/income, annuity/credit (`CREDIT_TERM`), goods price/credit,
  employment length/age, income per family member, and the mean and standard deviation of the external scores. Division
  by zero gives a missing value, never infinity.
- **Credit Bureau (`BUR_`):** number of credits, active credits and active share, total/mean credit amount, total debt
  and overdue amounts, maximum days overdue, mean credit age, debt-to-credit ratio, plus `HAS_BUREAU_HISTORY`.
- **Previous applications (`PREV_`):** counts and shares of approved/refused applications, mean requested and granted
  amounts, mean requested/granted ratio, days since the last decision, plus `HAS_PREV_APPLICATION`.
- **Instalments (`INST_`):** share of late instalments, mean/max days late, mean/min payment ratio, share of underpaid
  instalments and number of instalments. Partial payments of the same instalment are combined first: 8.90% of payment
  rows are below the instalment amount, but only 0.0426% of instalments are underpaid once the parts are summed. An
  extra feature, `INST_PARTIAL_PAYMENT_SHARE`, keeps the partial-payment behaviour.

Coverage: 94.39% of applicants without bureau history have a previous Home Credit application and 94.59% have
instalment history. Engineered behavioural features separate risk clearly: the default rate rises from 6.80%
(never late) to 15.43% (more than half of instalments late) and from 7.07% (no refusals) to 15.93% (more than half of
previous applications refused); `EXT_SOURCE_MEAN` deciles range from 22.93% to 2.04%
(`13_default_rate_by_inst_late_share.png`, `14_default_rate_by_prev_refused_share.png`,
`15_default_rate_by_ext_source_mean_decile.png`).

## 4 Exploratory data analysis

- **Imbalance:** 24,825 of 307,511 applicants (8.07%) defaulted, i.e. 11.39 repaid loans per default
  (`01_target_distribution.png`).
- **Age:** the default rate falls from 12.31% at age 20–24 to 3.72% at 65–69 (`03_default_rate_by_age_group.png`).
- **Socio-economic profile:** lower-secondary education 10.93% vs higher education 5.36%; working applicants 9.59% vs
  pensioners 5.39%; cash loans 8.35% vs revolving loans 5.48%; low-skill labourers 17.16% vs accountants 4.83%; civil
  marriage 9.94% vs widowed 5.82% (`04`–`08_default_rate_by_*.png`).
- **External scores:** `EXT_SOURCE_1/2/3` are the numeric columns most correlated with default (−0.1553, −0.1605,
  −0.1789), with clearly lower values for defaulters (`09_ext_source_kde_by_target.png`,
  `11_top15_target_correlations.png`).
- **Affordability:** credit-to-income and annuity-to-income ratios show only weak, non-monotonic relationships with
  default (at most 2.29 percentage points between deciles; `10_default_rate_by_affordability_deciles.png`).
- **Thin-file view:** 44,020 applicants (14.32%) have no Credit Bureau record and default more often (10.12% vs
  7.73%). In contrast, the 16,454 applicants (5.35%) without a previous Home Credit application default *less* often
  (5.96% vs 8.19%). Only 2,470 applicants (0.80%) have neither (`12_thin_file_groups.png`).

**Why do first-time Home Credit customers default less?** Three observations from the data point to selection
rather than to any benefit of being new. First, a first-time applicant is usually *not* thin-file: 84.99% of the 16,454
applicants without a previous Home Credit application have a Credit Bureau record, so most are established borrowers
who chose Home Credit for a particular purchase (17.56% of them take revolving loans, against 9.07% of returning
customers, and their median income is higher, 157,500 vs 144,000). The 2,470 first-time applicants who also lack a
bureau record default at 7.29%, close to the overall rate. Second, coming back is itself informative: the default
rate jumps from 5.96% with no previous application to 8.38% with one, stays around 7.7–8.1% for two to nine, and
reaches 9.68% for ten or more. Repeated demand for credit can signal financial strain. Third, rejection history
matters: among returning customers, the default rate is 7.07% when no earlier application was refused and 15.93%
when more than half were refused, consistent with riskier applicants re-applying after being turned down (adverse
selection). These are associations in observational data, not causal effects, and the "no previous application"
group is too heterogeneous to be treated as a single thin-file segment.

## 5 Clustering (K-Means)

**Goal.** Find natural groups of applicants without using the outcome, then check how their default rates differ
(notebook `05_clustering.ipynb`, code in `src/clustering.py`).

**Features and preprocessing.** Eight profile features: `AGE_YEARS`, log of `AMT_INCOME_TOTAL` (income is extremely
right-skewed), `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `YEARS_EMPLOYED`, `EXT_SOURCE_MEAN`, `INST_LATE_SHARE`
and `PREV_REFUSED_SHARE`. Median imputation and `StandardScaler` are fitted on the 246,005 training applicants only
(K-Means uses Euclidean distance, so unscaled features would dominate); holdout applicants are only assigned to the
fitted centres.

**Choosing k.** K-Means (`n_init = 10`, `random_state = 42`) was fitted for k = 2–10 (`39_kmeans_elbow_silhouette.png`).
The silhouette score, computed on a 20,000-applicant sample, is highest at k = 2 (0.1755), but those two halves default
at 7.25% and 8.45%, too similar to be useful. Among k ≥ 3 the best silhouette is at **k = 6** (0.1634), which is also
where the elbow curve bends (adding the 6th cluster reduces inertia by 107,958, the 7th by only 76,238). All silhouette
values are low (0.1296–0.1755): applicants form overlapping groups rather than sharply separated ones.

**Segments.** Each cluster is named after its most extreme standardised centre (`40_kmeans_cluster_heatmap.png`); the
risk level compares its training default rate with the overall 8.07% (low below 0.8×, high above 1.2×):

| Segment | Applicants (train) | Share | Default rate (train) | Default rate (holdout) | Mean age | Mean EXT_SOURCE_MEAN | Mean INST_LATE_SHARE | Mean PREV_REFUSED_SHARE |
|---|---|---|---|---|---|---|---|---|
| Long-tenure employees (low risk) | 20,938 | 8.5% | 3.81% | 4.10% | 48.3 | 0.569 | 0.061 | 0.076 |
| Older applicants with strong external scores (low risk) | 66,471 | 27.0% | 4.00% | 4.11% | 53.8 | 0.597 | 0.045 | 0.048 |
| High loan-to-income borrowers (medium risk) | 38,227 | 15.5% | 7.23% | 6.66% | 46.5 | 0.525 | 0.058 | 0.060 |
| Young, early-career applicants (high risk) | 69,644 | 28.3% | 10.07% | 10.19% | 33.4 | 0.435 | 0.049 | 0.034 |
| Frequent late payers (high risk) | 18,330 | 7.5% | 12.70% | 12.86% | 42.5 | 0.476 | 0.461 | 0.112 |
| Often refused before (high risk) | 32,395 | 13.2% | 13.27% | 13.12% | 41.3 | 0.450 | 0.073 | 0.461 |

Holdout default rates differ from the training rates by at most 0.56 percentage points, so the segments generalise
(`41_kmeans_default_rate.png`). The riskiest segments are defined by behaviour with the lender (late instalments,
refused applications), the safest by age, long employment and strong external scores. High loan-to-income borrowers
(mean credit/income 8.244 vs at most 3.816 in the other segments) default *below* average, consistent with the weak
affordability signal in the EDA. In a 2-D view (`42_kmeans_clusters_2d.png`), a 2-component PCA used only for plotting
shows 44.1% of the variance and separates the loan-to-income and age/tenure segments; the two most separating features,
`INST_LATE_SHARE` and `PREV_REFUSED_SHARE`, separate the two behavioural segments.

**Does the segment help prediction?** Adding the cluster label as a feature to the tuned LightGBM changes CV AUC from
0.7758 to 0.7763 and holdout AUC from 0.7819 to 0.7813 (−0.0006). The model uses the label (6th of 162 features by
gain) but it only re-packages information already in the raw features. The segments are therefore useful for
describing and communicating risk groups, not for improving accuracy.

<!-- INSIGHT START -->
In my view, the six segments are most useful as a simple map for business rules rather than for scoring. The two
low-risk segments together cover 35.53% of training applicants and default at only 3.95% combined, so a lender could
give them a faster, lighter approval process or pre-approved top-up offers and spend less review time on them. At the
other end, the two behaviour-based segments (frequent late payers and often refused before) cover 20.62% of applicants
but default at 13.07% combined, more than three times the low-risk rate; these customers could start with smaller
limits, receive payment reminders before each due date and be contacted earlier by collections. The largest segment,
young early-career applicants (28.3% of applicants, 10.07% default), is also the group a lender needs for future
growth, so instead of simply rejecting them it could offer starter loans with low limits that rise after on-time
repayments.
<!-- INSIGHT END -->

## 6 Machine learning models and justification

All five models are trained on the same 246,005 training applicants and 161 features, tuned with cross-validation on
the training set only, and evaluated once on the 61,502 holdout applicants (notebook `03_modeling.ipynb`, code in
`src/modeling.py`).

| Model | How it works | Why it is in the comparison | Preprocessing | Final settings and tuning | Imbalance handling |
|---|---|---|---|---|---|
| Logistic regression | A weighted sum of the inputs passed through the logistic function | The industry-standard credit scorecard and the baseline; transparent coefficients | Median imputation + missing indicators, `StandardScaler`, one-hot (325 inputs) | `max_iter = 2000`, L2 penalty (default); not tuned | `class_weight = 'balanced'` |
| Decision tree | A sequence of yes/no questions on one feature at a time, chosen to separate defaulters from good applicants (Breiman et al., 1984) | The most interpretable non-linear model; the building block of the next three | Median imputation, one-hot (234 inputs); no scaling | Grid search, 3-fold CV: `max_depth` {4, 6, 8, 10, 12} × `min_samples_leaf` {50, 100, 200} → depth 6, leaves of 200 (64 leaves) | `class_weight = 'balanced'` |
| Random forest | Hundreds of deep trees, each on a random half of the applicants and a random subset of features; their votes are averaged (Breiman, 2001) | Bagging: averaging reduces the variance of single trees | As the decision tree | Time-bounded random search, 3-fold CV, 12 of 36 combinations of `n_estimators` {200, 400}, `max_depth` {12, 16, None}, `min_samples_leaf` {20, 50, 100}, `max_features` {'sqrt', 0.3} → 400 trees, no depth limit, leaves of 20, `max_features` 0.3; `max_samples` 0.5 | `class_weight = 'balanced_subsample'` |
| XGBoost | Gradient boosting: shallow trees added one at a time, each fitted to the errors of the current ensemble (Chen & Guestrin, 2016) | A second boosting implementation; native missing values and categoricals | None needed (native) | Fixed, untuned: learning rate 0.05, depth 6, `min_child_weight` 5, subsample 0.8, column sample 0.5, `reg_lambda` 1; early stopping → 333 trees | `scale_pos_weight` = negatives/positives |
| LightGBM | Gradient boosting with leaf-wise trees and histogram binning (Ke et al., 2017) | State of the art for large tabular data; fast; the main model | None needed (native) | Random search of 20 configurations, 5-fold CV → `num_leaves` 15, learning rate 0.02, `min_child_samples` 100, feature fraction 0.3, bagging fraction 0.8, `reg_lambda` 10; early stopping → 2,203 trees | `scale_pos_weight` = negatives/positives |

**Common protocol.** Every model is scored with the same stratified 5-fold CV on the training set (`random_state = 42`);
its out-of-fold predictions are kept to choose its decision threshold. The boosting models stop adding trees when AUC
on a stratified 10% carve-out of the data being fitted has not improved for 100 rounds, so the fold being scored is
never used for stopping. Imbalance is handled by class weighting rather than resampling: it keeps every applicant and
makes the rare defaults count about 11 times as much during training. Training time of the final fit on the full
training set: decision tree 16.8 s, LightGBM 17.9 s, logistic regression 18.7 s, XGBoost 19.3 s, random forest 99.8 s.
The final tree's top three levels are drawn in `33_decision_tree_top3_levels.png`: every split in the top two levels
uses `EXT_SOURCE_MEAN`, which accounts for 81.11% of the tree's impurity reduction.

Together the five models span the bias–variance spectrum: a linear model, a single tree, bagging (variance reduction)
and boosting (bias reduction with regularisation), which is what section 9 examines.

## 7 Evaluation metrics and justification

With 8.07% defaults, a model that approves everyone is 91.93% accurate on the holdout yet catches no defaulter, so
accuracy cannot be the main criterion. Nine metrics are reported for every model (`reports/metrics_table_final.csv`):

| Metric | What it measures | Why it matters here |
|---|---|---|
| AUC-ROC | Probability that a random defaulter gets a higher score than a random good applicant | Threshold-free and unaffected by the class ratio; measures ranking, which any cut-off relies on |
| Gini (= 2·AUC − 1) | The same, on the scale used by credit-risk teams | Industry convention |
| PR-AUC (average precision) | Precision against recall for the default class | Its random baseline equals the default rate (0.0807), so it is the stricter view when positives are rare (Saito & Rehmsmeier, 2015) |
| KS statistic | Largest gap between the cumulative score distributions of defaulters and good applicants | Classic scorecard separation measure |
| Recall (default class) | Share of defaulters the model rejects | Missed defaulters are the expensive error (loss given default) |
| Precision (default class) | Share of rejected applicants who really default | Every false alarm is a lost customer |
| F1 (default class) | Harmonic mean of precision and recall | One number balancing the two errors, ignoring the many easy true negatives |
| Accuracy | Share of correct decisions | Reported to show why it misleads under imbalance |
| Log loss | Penalty for confident wrong probabilities | Judges scores as probabilities; class weighting deliberately pushes scores above real default rates, which inflates log loss, so it is reported but not used to choose a model |

Threshold-dependent metrics are reported at two thresholds: the conventional 0.5 and each model's **cost-optimal
threshold**, chosen on its training out-of-fold predictions to minimise 0.60 × `AMT_CREDIT` per approved defaulter plus
0.15 × `AMT_CREDIT` per rejected good applicant (`src/cost.py`). An applicant is rejected when the score is at or above
the threshold.

## 8 Results and comparative analysis

**Threshold-free metrics and threshold 0.5** (holdout, 61,502 applicants; best value per column in bold):

| Model | CV AUC (5-fold, train) | Holdout AUC | Gini | PR-AUC | KS | Log loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.7600 ± 0.0013 | 0.7641 | 0.5281 | 0.2482 | 0.3989 | 0.5778 | 0.6997 | 0.1694 | **0.6971** | 0.2726 |
| Decision tree | 0.7266 ± 0.0022 | 0.7269 | 0.4539 | 0.1988 | 0.3346 | 0.6032 | 0.6507 | 0.1462 | 0.6870 | 0.2410 |
| Random forest | 0.7592 ± 0.0022 | 0.7618 | 0.5236 | 0.2495 | 0.3987 | **0.3601** | **0.8752** | **0.2753** | 0.3347 | **0.3021** |
| XGBoost | 0.7698 ± 0.0017 | 0.7744 | 0.5489 | 0.2693 | 0.4079 | 0.5045 | 0.7556 | 0.1931 | 0.6377 | 0.2964 |
| **LightGBM** | **0.7758 ± 0.0015** | **0.7819** | **0.5638** | **0.2818** | **0.4237** | 0.5194 | 0.7414 | 0.1896 | 0.6733 | 0.2959 |

**Each model's cost-optimal threshold** (chosen on training out-of-fold predictions):

| Model | Threshold | Accuracy | Precision | Recall | F1 | Approval rate | Default rate among approved | Saving vs approve all |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.74 | 0.8832 | 0.2894 | 0.3072 | 0.2980 | 91.43% | 6.12% | 10.74% |
| Decision tree | 0.76 | **0.8950** | 0.2825 | 0.1950 | 0.2307 | 94.43% | 6.88% | 6.72% |
| Random forest | 0.52 | 0.8828 | 0.2877 | 0.3065 | 0.2968 | 91.40% | 6.12% | 10.82% |
| XGBoost | 0.70 | 0.8805 | 0.2892 | 0.3293 | 0.3079 | 90.81% | 5.96% | 11.65% |
| **LightGBM** | 0.70 | 0.8750 | **0.2903** | **0.3797** | **0.3290** | 89.44% | **5.60%** | **14.04%** |

**Ranking quality.** The holdout ranking is the CV ranking: LightGBM (AUC 0.7819) > XGBoost (0.7744) > logistic
regression (0.7641) > random forest (0.7618) > decision tree (0.7269), with fold-to-fold spreads of at most 0.0022. The
ROC and precision-recall curves show the same order (`16_roc_curves.png`, `17_pr_curves.png`); the random forest edges
past logistic regression on PR-AUC (0.2495 vs 0.2482), i.e. at the riskiest end of the ranking. The decision tree's
64 leaves give only 64 distinct scores, which limits its ranking.

**How accuracy and the other metrics change across models** (`35_metrics_grouped_bars.png`, `36_metrics_heatmap.png`).
At 0.5, accuracy ranges from 0.6507 (decision tree) to 0.8752 (random forest) and recall from 0.3347 (random forest)
to 0.6971 (logistic regression). These swings reflect how many applicants each model flags at the same nominal
cut-off, not how well it ranks: the random forest's scores are the least inflated (mean holdout score 0.2555 vs
0.3638–0.4273 for the others), so it flags far fewer applicants at 0.5. At the cost-optimal thresholds accuracy
converges (0.8750–0.8950) and LightGBM leads on precision, recall and F1. The decision tree is the most "accurate"
there only because it rejects the fewest applicants and catches 19.50% of defaulters. The threshold curves
(`37_metrics_vs_threshold.png`) make this explicit: raising the threshold lowers recall and raises accuracy towards
the 91.93% of approving everyone, and F1 peaks at an intermediate threshold for every model (LightGBM 0.3304 at 0.66).

**Decisions.** At the cost-optimal thresholds LightGBM catches 1,885 of the 4,965 holdout defaulters while rejecting
4,608 good applicants and saves 14.04% vs approving everyone, ahead of XGBoost (11.65%), random forest (10.82%),
logistic regression (10.74%) and the decision tree (6.72%) (`34_confusion_matrices_cost_optimal.png`). At 0.5, four of
the five models cost more than approving everyone (from −3.91% for XGBoost to −32.33% for the decision tree,
`20_confusion_matrices.png`); only the random forest, whose optimum (0.52) is close to 0.5, saves money (10.61%).

**Probabilities.** No model outputs real default probabilities: mean holdout scores are 0.2555–0.4273 against an actual
rate of 0.0807, and even the top-score decile defaults at only 24.33%–29.36% (`19_calibration_curves.png`). This is the
intended effect of class weighting and explains the high log losses; it does not affect ranking.

**Choice of model.** LightGBM is selected: it is best on every ranking metric, on precision, recall and F1 at its
cost-optimal threshold, and on cost, trains in 17.9 s, and handles missing values and categorical features natively.
The leakage guard was never triggered (maximum holdout AUC 0.7819).

**Is an AUC of about 0.78 good enough?** On the same competition data, leading Kaggle teams that used all seven
tables, extensive feature engineering and model ensembles reached private-leaderboard AUCs of about 0.80 (for
example 0.80265 for 17th place of 7,198; NoxMoon, 2018). Those scores come from a different, unlabelled test set, so
the comparison is indicative only, but it puts a single model built on four tables within about 0.02 AUC of
competition-level performance. The gain over logistic regression (+0.0179 AUC, +0.0357 Gini) is modest, which is
consistent with Lessmann et al. (2015), who found that advanced ensembles beat logistic regression on most credit
scoring datasets but by small margins. AUC alone does not decide whether a model is good enough, because it measures
ranking, not decisions. The relevant test is economic: section 10.1 shows that this ranking, combined with a sensible
threshold, lowers expected cost by 14.04% relative to approving everyone. The model is therefore useful as a decision
input, but not as a fully automatic decision-maker. At the chosen threshold it still approves 3,080 holdout
defaulters, and its scores must be calibrated before they can be read as probabilities. A regulated lender could
also reasonably prefer the slightly weaker but more transparent scorecard. SHAP reason codes (section 10.3) narrow
that transparency gap.

<!-- INSIGHT START -->
The result that surprised me most was the random forest. With 400 trees, 99.8 seconds of training and a 57 MB model
file, I expected it to beat the simple logistic regression, but its holdout AUC was slightly lower (0.7618 vs 0.7641),
and at the cost-optimal thresholds the two saved almost the same amount (10.82% vs 10.74%). This shows that a more
complex model is not automatically a better one: the forest's extra capacity mostly went into memorising the training
data rather than finding new patterns. For a regulated lender I would still keep LightGBM as the decision model,
because switching to the scorecard would give up 3.30 percentage points of saving, which is 54,982,702 currency units
on the 61,502 holdout applicants alone. I would, however, run the logistic regression alongside it as a benchmark, so
that applicants on whom the two models strongly disagree can be sent for manual review.
<!-- INSIGHT END -->

## 9 Bias-variance discussion

A model with **high bias** is too simple for the pattern in the data: it scores poorly on its own training data and on
new data alike. A model with **high variance** fits its training sample too closely: it scores much better on the data
it was trained on than on new data. The check below compares each final model's AUC on its own training data
(in-sample) with its 5-fold CV and holdout AUC (`reports/overfitting_check.csv`, `38_overfitting_check.png`).

| Model | Train AUC | CV AUC | Holdout AUC | Train − CV | Holdout − CV |
|---|---|---|---|---|---|
| Logistic regression | 0.7642 | 0.7600 | 0.7641 | +0.0042 | +0.0041 |
| Decision tree | 0.7388 | 0.7266 | 0.7269 | +0.0122 | +0.0003 |
| Random forest | 0.9353 | 0.7592 | 0.7618 | +0.1762 | +0.0026 |
| XGBoost | 0.8628 | 0.7698 | 0.7744 | +0.0930 | +0.0046 |
| LightGBM | 0.8433 | 0.7758 | 0.7819 | +0.0675 | +0.0061 |

- **Logistic regression and the shallow tree: high bias.** Their train and CV AUCs are almost equal, but both stay well
  below the boosting models (0.7600 and 0.7266 vs 0.7758): a linear boundary and a depth-6 tree cannot represent the
  interactions in the data.
- **Deeper single trees: high variance.** In the decision-tree grid (3-fold CV, right panel of the figure), the
  training AUC rises steadily with depth while the validation AUC peaks at depth 6 and then falls; at depth 12 with
  leaves of 50 the training AUC is 0.8295 but the CV AUC only 0.6792.
- **Bagging reduces variance.** The random forest's trees are grown without a depth limit and memorise the training
  data (train AUC 0.9353), yet averaging 400 of them, each on a different half of the applicants and feature subsets,
  lifts CV AUC to 0.7592, far above any single tree.
- **Boosting with regularisation balances both.** XGBoost and LightGBM reduce bias by adding trees that correct earlier
  errors, and control variance with small trees (`num_leaves` 15), L2 regularisation (`reg_lambda` 10), feature and
  row sampling and early stopping. They have moderate train–CV gaps and the best CV and holdout AUC.

A large train–CV gap is not by itself a failure: what matters for new applicants is the CV and holdout performance,
and the holdout AUC lies within +0.0003 to +0.0061 of the CV AUC for every model, so CV was a reliable guide and no
model's performance collapsed on unseen data.

<!-- INSIGHT START -->
If I had more time, I would work on bias before variance. The tuning results suggest that the models are limited by
the information in the 161 features rather than by overfitting: twenty LightGBM settings only moved the CV AUC between
0.7641 and 0.7758, and the twelve random-forest settings only between 0.7552 and 0.7584. So my first change would be to
add the three monthly behavioural tables that were left out (bureau balance, point-of-sale and credit-card histories),
because new information can lower bias in a way that more tuning cannot, and repayment behaviour was already one of
the strongest signals for thin-file applicants. For variance, my first target would be XGBoost, which was never tuned
and has a larger train–CV gap than LightGBM (0.0930 vs 0.0675) while using a much weaker L2 penalty (`reg_lambda` 1 vs
10), so stronger regularisation should help it generalise better.
<!-- INSIGHT END -->

## 10 Cost-based threshold, thin-file analysis and explainability (SHAP)

This section uses the final LightGBM model (notebook `04_threshold_explainability.ipynb`).

### 10.1 Cost-based decision threshold

Approving a defaulter is assumed to cost LGD × credit amount (LGD = 0.60), and rejecting a good customer MARGIN ×
credit amount (MARGIN = 0.15). The threshold is chosen on the training out-of-fold predictions; the result, **0.70**,
coincides with the threshold that would minimise holdout cost (computed only as a check).

| Holdout policy | Approval rate | Default rate among approved | Total cost | vs approve everyone |
|---|---|---|---|---|
| Approve everyone | 100.00% | 8.07% | 1,665,214,629 | — |
| Threshold 0.50 | 71.34% | 3.70% | 1,756,164,150 | −5.46% |
| Threshold 0.70 | 89.44% | 5.60% | 1,431,358,554 | +14.04% saving |

The conventional 0.5 cut-off is worse than approving everyone because it rejects 14,285 good applicants. The
optimum lies far above the break-even of 0.20 that calibrated probabilities would imply, again reflecting the inflated
scores of the class-weighted model. In a sensitivity grid (LGD 0.4/0.6/0.8 × MARGIN 0.10/0.15/0.20), the optimal
threshold ranges from 0.57 to 0.83 and the saving from 4.53% to 30.04%; only the ratio MARGIN/LGD matters, and every
scenario beats approving everyone (`21_cost_vs_threshold.png`, `22_approval_vs_default_rate.png`,
`23_threshold_sensitivity.png`). These cost parameters are assumptions, not data.

**Which cost values are realistic?** For loss given default, a regulatory anchor is the 45% supervisory LGD that the
Basel II foundation approach assigns to senior unsecured claims on corporates, banks and sovereigns (Basel Committee
on Banking Supervision, 2006). Retail lenders must estimate their own LGD, and unsecured consumer loans to
low-income borrowers in emerging markets usually recover less than secured or corporate debt, so 0.60 is a plausible
central value and the 0.40–0.80 range a reasonable bracket. The margin is harder to pin down. It should be the
*net* profit lost by rejecting a good customer (interest income minus funding, operating and expected-loss costs) over
the life of the loan, plus the value of future business with that customer. For a short point-of-sale loan it may be
well below 0.15 of the credit amount. For a multi-year cash loan at the high interest rates typical of this market,
or once repeat business is counted, it may exceed 0.20. Because only the ratio MARGIN/LGD matters, the recommendation
is robust in one respect and not in another. Every scenario in the grid beats approving everyone, and every optimal
threshold (0.57–0.83) is above the conventional 0.5, so 0.5 is too strict under any of these assumptions. The exact
cut-off, however, moves with the cost ratio. A lender should therefore estimate LGD from its own recovery data and
margin from product pricing, re-run the threshold selection on out-of-fold predictions (the Streamlit app does this
interactively), and consider separate thresholds for cash and revolving products, whose costs differ.

### 10.2 Thin-file analysis

On the holdout, LightGBM's AUC is 0.7645 (95% bootstrap interval 0.7512–0.7794) for the 8,755 applicants without
bureau history versus 0.7837 for those with it; logistic regression falls further (0.7348 vs 0.7678)
(`24_auc_by_history_group.png`). Ablation models with the same hyperparameters (`25_ablation_auc.png`) show:

- removing the external scores lowers overall AUC from 0.7819 to 0.7548;
- removing the bureau features barely changes it (0.7783), and not at all for bureau-less applicants (0.7654);
- using **alternative data only** (no external scores, no bureau data) gives AUC 0.7373 overall and **0.7507** for
  bureau-less applicants, only 0.0138 below the full model on that group, and better than for applicants with a
  bureau file (0.7331).

For bureau-less applicants the alternative-data model relies mainly on loan structure (`CREDIT_TERM`,
`GOODS_CREDIT_RATIO`, `PREV_APP_CREDIT_RATIO_MEAN`), repayment behaviour on earlier Home Credit loans
(`INST_LATE_SHARE`), stability indicators (`DAYS_LAST_PHONE_CHANGE`, `ORGANIZATION_TYPE`), and gender
(`31_shap_beeswarm_no_bureau.png`). The answer to RQ2 is therefore positive: alternative data can rank thin-file
applicants with useful accuracy. But it depends heavily on the lender's own history with the customer, and only 2,470
applicants (0.80%) have neither bureau nor previous-application data.

**Implications for genuinely new-to-credit customers.** The thin-file result relies on Home Credit's *own* history
with the customer: repayment behaviour on earlier loans (`INST_LATE_SHARE`) and the structure of earlier applications
are among the strongest signals for bureau-less applicants. A lender with no previous relationship has none of this.
The data say little about such customers: the holdout contains only 480 applicants with neither bureau nor
previous-application data (35 defaults). LightGBM ranks them with AUC 0.7893, but the 95% bootstrap interval
(0.7295–0.8564) is too wide to support any firm conclusion. Three implications follow. First, such a lender needs
alternative data from *outside* the credit relationship. Berg et al. (2020) show that simple digital-footprint
variables predict default about as well as a credit-bureau score and complement it, and Óskarsdóttir et al. (2019)
report gains from mobile-phone and social-network data, which raise privacy and consent questions of their own.
Second, the lender can build its own history on purpose, starting new customers on small loans or low limits that
grow with on-time repayment, since repayment behaviour becomes one of the strongest signals once it exists. Third,
application-form information alone should be validated separately before it is relied on for this group. That model
was not built here and is natural future work.

### 10.3 Explainability (SHAP)

**Global drivers.** By mean |SHAP| on 5,000 holdout applicants, the top drivers are `EXT_SOURCE_MEAN` (0.3610),
`CREDIT_TERM` (0.1265), `PREV_APP_CREDIT_RATIO_MEAN` (0.1208), `GOODS_CREDIT_RATIO` (0.1160), `ORGANIZATION_TYPE`
(0.1124), `CODE_GENDER` (0.1095) and `INST_LATE_SHARE` (0.1007) (`26_shap_beeswarm.png`, `27_shap_bar.png`).

**Local explanations.** For three applicants rejected at threshold 0.70, the largest risk-increasing factors were low
external credit scores (averages 0.17, 0.13 and 0.30), outstanding bureau debt at 99% of bureau credit, and the
requested-to-granted ratio on previous loans; these were turned into plain-English reason codes
(`28`–`30_shap_waterfall_applicant_*.png`). Two of the three actually repaid, illustrating the good customers a
cost-optimal policy accepts losing.

## 11 Fairness considerations

**Fairness.** At threshold 0.70 women are approved more often than men (91.95% vs 84.60%), consistent with their
lower actual default rate (6.99% vs 10.16%), but good male applicants are rejected 1.92 times as often (11.97% vs
6.25%). Approval rises with age from 79.30% (20–29) to 97.23% (60–69) (`32_fairness_gender_age.png`). Retraining
without `CODE_GENDER` reduces holdout AUC only from 0.7819 to 0.7803, raises holdout cost by 0.94% and narrows the
approval gap from 7.35 to 5.10 percentage points. The gap does not vanish because gender can be predicted from the
other features with AUC 0.9094 (mainly through `OCCUPATION_TYPE`), so "fairness through unawareness" is only a partial
remedy.

**Which fairness definition fits, and what does the law allow?** *Demographic parity* (equal approval rates) ignores
the real difference in repayment between groups (6.99% vs 10.16% default). Enforcing it would mean approving riskier
men or rejecting creditworthy women, so it fits poorly unless the gap itself reflects historical disadvantage that
the lender intends to correct. *Equal opportunity* (Hardt et al., 2016) asks that applicants who would repay be
approved at the same rate in every group. This matches the lending principle that creditworthy people should have
equal access to credit, and it is clearly violated here: 11.97% of good male applicants are rejected against 6.25% of
good female applicants. *Calibration within groups*, meaning that a given score implies the same default risk for
everyone, is needed for risk-based pricing and for the lender's own solvency. When base rates differ, a non-trivial
classifier cannot satisfy both calibration and equal error rates (Kleinberg et al., 2017; Chouldechova, 2017), so the
lender must choose. A defensible policy is to require calibration within groups and to monitor the equal-opportunity
gap against an agreed tolerance.

Legally, the country of the data is undisclosed, so only examples can be given. In the United States, the Equal
Credit Opportunity Act and Regulation B (12 CFR Part 1002) prohibit discrimination on the basis of sex and marital
status. An empirically derived, statistically sound scoring system may use age only if applicants aged 62 or older
are not assigned a negative value; in this model, older applicants are in fact approved more often. In the European
Union, Council Directive 2004/113/EC prohibits sex discrimination in access to goods and services, including
financial services. The AI Act (Regulation (EU) 2024/1689) classifies AI systems that evaluate the creditworthiness
of natural persons as high-risk, which requires risk management, data governance, human oversight and
documentation. Under rules like these, `CODE_GENDER` should not be a model input. Because gender can be predicted
from the remaining features with AUC 0.9094, removing it does not remove the risk of indirect discrimination, so
outcomes still need disparate-impact testing.

## 12 Conclusion

Five models were compared under one leakage-safe protocol. LightGBM is the best model on every ranking metric (holdout
AUC 0.7819, Gini 0.5638, PR-AUC 0.2818, KS 0.4237) and, at its cost-optimal threshold, on precision (0.2903), recall
(0.3797) and F1 (0.3290); XGBoost is second, logistic regression and the random forest are close behind, and a single
decision tree is clearly weaker (AUC 0.7269). Accuracy turned out to be the least informative metric: it is highest for
the models that reject the fewest applicants, and approving everyone would beat every model on it. The bias–variance
check explains the ranking: the linear model and shallow tree underfit, unrestricted trees overfit, and bagging and
regularised boosting control the variance, with holdout AUC within 0.0061 of CV for every model. K-Means found six
interpretable applicant segments with default rates from 3.81% to 13.27%; behaviour with the lender (late instalments,
refusals) marks the riskiest groups, but the segment label adds nothing to LightGBM's accuracy.

For the research questions: alternative data can predict loan default (RQ1), and for applicants without bureau history
LightGBM still reaches AUC 0.7645, or 0.7507 without any bureau data or external scores (RQ2). Translating scores into
decisions matters as much as the model: a cost-based threshold of 0.70 saves 14.04% of expected cost compared with
approving everyone, whereas the default 0.5 cut-off loses money for four of the five models (RQ3). Rejections are
driven mainly by low external scores, loan structure and past repayment behaviour, but the model also uses gender,
and removing the variable only partly closes the approval gap because other features act as proxies (RQ4).
Responsible deployment would therefore need calibrated scores, validated cost assumptions and ongoing fairness
monitoring.

**Recommendation.** For a lender serving thin-file customers, the most valuable alternative data in this study is
the lender's own relationship with the applicant: earlier applications and repayment behaviour carry most of the
signal once bureau data are missing. The recommended policy is:

1. use the LightGBM model *without* `CODE_GENDER`, which costs only 0.0016 AUC and 0.94% of holdout cost;
2. calibrate its scores on out-of-fold predictions and set the cut-off from validated LGD and margin estimates
   (about 0.70 under the assumptions used here), never the default 0.5;
3. give every rejected applicant SHAP-based reason codes;
4. monitor the equal-opportunity gap by gender and age, and the score distribution, after deployment;
5. for applicants with no history at all, add external alternative data and start with small, growing limits rather
   than relying on this model.

Choosing the decision rule matters as much as choosing the model: the same scores lose money at 0.5 and save 14% at
0.70.

<!-- INSIGHT START -->
Looking back, the most important thing I learned is that a result is only worth something if it can be trusted and
reproduced. Because the train/holdout split was fixed once and `random_state = 42` was used everywhere, re-running the
whole modelling notebook gave exactly the same holdout predictions for the three original models (maximum difference 0),
so every number in this report can be checked. I also found it useful to study a simple model before trusting a complex
one: every split in the top two levels of the decision tree used `EXT_SOURCE_MEAN` (81.11% of its impurity reduction),
and SHAP later confirmed the same feature as LightGBM's strongest driver (mean |SHAP| 0.3610), so the black-box model
was telling the same story as the readable one. Finally, the project taught me to be honest about what the data cannot
show: only 480 holdout applicants had no credit history of any kind, so I cannot claim the model works for truly
new-to-credit customers, even though they are the people this project set out to help.
<!-- INSIGHT END -->

## 13 Limitations and future scope

- **Undisclosed context:** the country, currency and exact target definition (X days, Y instalments) are not
  disclosed, which limits economic interpretation of the cost results.
- **Assumed costs:** LGD and margin are assumptions; real values vary by product and over time.
- **Uncalibrated scores:** class weighting inflates probabilities; a calibration step (e.g. isotonic regression on
  out-of-fold predictions) would be needed before using scores as probabilities of default.
- **Selection bias:** outcomes exist only for loans that were granted; rejected applicants are not observed (reject
  inference was not attempted).
- **No time dimension:** the data contain no application dates, so out-of-time validation was not possible.
- **Data scope:** three behavioural tables (`bureau_balance`, `POS_CASH_balance`, `credit_card_balance`) were
  excluded; they may add signal, particularly for thin-file applicants.
- **Thin-file definition:** "no bureau record" applicants mostly still have Home Credit history; truly new-to-credit
  applicants (2,470, 0.80%) are too few for a separate evaluation.
- **Analysis choices:** EDA used all labelled data before the split (nothing was fitted on it); the hyperparameter
  searches were small (20 LightGBM configurations, 12 of 36 random-forest combinations, a 15-point tree grid) and
  XGBoost was not tuned; a single random holdout was used; fairness was checked only for gender and age.
- **Clustering:** K-Means assumes roughly spherical clusters of similar size; silhouette scores are low (0.1634 at
  k = 6), so the segments are useful summaries rather than sharply separated groups.
- **Future work:** add the excluded behavioural tables; calibrate scores; tune XGBoost and widen the searches; test fairness-aware training or
  post-processing; evaluate stability across segments and over time if dated data become available.

## References

- Home Credit Group (2018). *Home Credit Default Risk* [Kaggle competition dataset].
  https://www.kaggle.com/competitions/home-credit-default-risk
- Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning: Limitations and Opportunities.* MIT Press.
- Berg, T., Burg, V., Gombović, A., & Puri, M. (2020). On the rise of FinTechs: Credit scoring using digital footprints.
  *The Review of Financial Studies*, 33(7), 2845–2897.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees.*
  Wadsworth.
- Brevoort, K. P., Grimm, P., & Kambara, M. (2015). *Data Point: Credit Invisibles.* Consumer Financial Protection
  Bureau, Office of Research.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD
  Conference on Knowledge Discovery and Data Mining*, 785–794.
- Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction
  instruments. *Big Data*, 5(2), 153–163.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *Proceedings of IJCAI 2001*, 973–977.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural
  Information Processing Systems 29*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Kambara, M., & Luce, C. (2025). *Technical Correction and Update to the CFPB's Credit Invisibles Estimate.*
  Consumer Financial Protection Bureau, Office of Research.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and
  avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient
  gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
- Klapper, L., Singer, D., Starita, L., & Norris, A. (2025). *The Global Findex Database 2025: Connectivity and
  Financial Inclusion in the Digital Economy.* World Bank.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk
  scores. *Proceedings of the 8th Innovations in Theoretical Computer Science Conference (ITCS 2017).*
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification
  algorithms for credit scoring: An update of research. *European Journal of Operational Research*, 247(1), 124–136.
- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129–137.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural
  Information Processing Systems 30*.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global understanding with
  explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of
  the 22nd International Conference on Machine Learning*, 625–632.
- NoxMoon (2018). *home-credit-default-risk* [GitHub repository; Kaggle private leaderboard 0.80265, rank 17 of
  7,198]. https://github.com/NoxMoon/home-credit-default-risk
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12,
  2825–2830.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.
  *Journal of Computational and Applied Mathematics*, 20, 53–65.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating
  binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3), e0118432.
- Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2nd ed.).
  Wiley.
- Óskarsdóttir, M., Bravo, C., Sarraute, C., Vanthienen, J., & Baesens, B. (2019). The value of big data for credit
  scoring: Enhancing financial inclusion using mobile phone data and social network analytics. *Applied Soft
  Computing*, 74, 26–39.

**Regulatory and legal sources**

- Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital
  Standards: A Revised Framework — Comprehensive Version.* Bank for International Settlements.
- Council Directive 2004/113/EC of 13 December 2004 implementing the principle of equal treatment between men and
  women in the access to and supply of goods and services. *Official Journal of the European Union*, L 373.
- Equal Credit Opportunity Act (Regulation B), 12 CFR Part 1002 (United States).
- Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised
  rules on artificial intelligence (Artificial Intelligence Act).
