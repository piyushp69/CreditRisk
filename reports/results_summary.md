# Results summary: predicting loan default for thin-file borrowers

**Data:** Home Credit Default Risk (official Kaggle competition, Home Credit Group): 307,507 labelled applicants after
cleaning, 8.07% default rate. Stratified 80/20 split: 246,005 train / 61,502 holdout (`data/processed/split_ids.parquet`).
All tuning, early stopping, clustering and threshold selection used the training set only; every model was evaluated
once on the holdout. Every number below comes from the notebook outputs (`notebooks/03_modeling.ipynb`,
`notebooks/04_threshold_explainability.ipynb`, `notebooks/05_clustering.ipynb`) and the files they save.

## 1. Five-model comparison (holdout, evaluated once)

**Threshold-free metrics and threshold 0.5** (full table: `reports/metrics_table_final.csv`, one row per model per
threshold type; best value per column in bold):

| Model | CV AUC (5-fold, train) | Training time (s) | Holdout AUC | Gini | PR-AUC | KS | Log loss | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.7600 ± 0.0013 | 18.7 | 0.7641 | 0.5281 | 0.2482 | 0.3989 | 0.5778 | 0.6997 | 0.1694 | **0.6971** | 0.2726 |
| Decision tree | 0.7266 ± 0.0022 | 16.8 | 0.7269 | 0.4539 | 0.1988 | 0.3346 | 0.6032 | 0.6507 | 0.1462 | 0.6870 | 0.2410 |
| Random forest | 0.7592 ± 0.0022 | 99.8 | 0.7618 | 0.5236 | 0.2495 | 0.3987 | **0.3601** | **0.8752** | **0.2753** | 0.3347 | **0.3021** |
| XGBoost | 0.7698 ± 0.0017 | 19.3 | 0.7744 | 0.5489 | 0.2693 | 0.4079 | 0.5045 | 0.7556 | 0.1931 | 0.6377 | 0.2964 |
| **LightGBM** | **0.7758 ± 0.0015** | 17.9 | **0.7819** | **0.5638** | **0.2818** | **0.4237** | 0.5194 | 0.7414 | 0.1896 | 0.6733 | 0.2959 |

**Each model's cost-optimal threshold** (chosen on its training out-of-fold predictions; FN cost 0.60 × `AMT_CREDIT`,
FP cost 0.15 × `AMT_CREDIT`, both assumptions):

| Model | Threshold | Accuracy | Precision | Recall | F1 | Defaulters caught | Good applicants rejected | Approval rate | Default rate among approved | Saving vs approve all |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.74 | 0.8832 | 0.2894 | 0.3072 | 0.2980 | 1,525 of 4,965 | 3,744 | 91.43% | 6.12% | 10.74% |
| Decision tree | 0.76 | **0.8950** | 0.2825 | 0.1950 | 0.2307 | 968 of 4,965 | 2,458 | 94.43% | 6.88% | 6.72% |
| Random forest | 0.52 | 0.8828 | 0.2877 | 0.3065 | 0.2968 | 1,522 of 4,965 | 3,768 | 91.40% | 6.12% | 10.82% |
| XGBoost | 0.70 | 0.8805 | 0.2892 | 0.3293 | 0.3079 | 1,635 of 4,965 | 4,019 | 90.81% | 5.96% | 11.65% |
| **LightGBM** | 0.70 | 0.8750 | **0.2903** | **0.3797** | **0.3290** | **1,885 of 4,965** | 4,608 | 89.44% | **5.60%** | **14.04%** |

**Best model per metric.** LightGBM wins every threshold-free metric (AUC, Gini, PR-AUC, KS), precision, recall and F1
at the cost-optimal thresholds, and the largest cost saving. The other winners depend on how many applicants a model
happens to flag: random forest (accuracy, precision and F1 at 0.5; lowest log loss because its scores are the least
inflated), logistic regression (recall at 0.5) and decision tree (accuracy at its cost-optimal threshold, because it
rejects the fewest applicants and catches only 19.50% of defaulters).

**Why accuracy misleads:** approving everyone is 91.93% accurate on the holdout and catches no defaulter. At 0.5 four
of the five models cost *more* than approving everyone (logistic regression −17.52%, decision tree −32.33%,
XGBoost −3.91%, LightGBM −5.46%); only the random forest, whose cost-optimal threshold (0.52) is close to 0.5, saves
money there (10.61%). Class weighting makes all scores higher than real default probabilities (mean holdout score
0.2555–0.4273 vs an actual rate of 0.0807), which also inflates log loss.

Tuned settings: decision tree `max_depth` 6, `min_samples_leaf` 200 (grid of 15, 3-fold CV; 64 leaves); random forest
400 trees, `max_depth` None, `min_samples_leaf` 20, `max_features` 0.3 (12 of 36 combinations sampled, 3-fold CV,
19.4 min); LightGBM `num_leaves` 15, `learning_rate` 0.02, `min_child_samples` 100, `colsample_bytree` 0.3,
`subsample` 0.8, `reg_lambda` 10 (20-trial random search, 5-fold CV; 2,203 trees); XGBoost fixed parameters
(`learning_rate` 0.05, `max_depth` 6; 333 trees). No holdout AUC exceeds the 0.85 leakage alarm (maximum 0.7819).
Figures: `16_roc_curves.png`, `17_pr_curves.png`, `19_calibration_curves.png`, `20_confusion_matrices.png`,
`33_decision_tree_top3_levels.png`, `34_confusion_matrices_cost_optimal.png`, `35_metrics_grouped_bars.png`,
`36_metrics_heatmap.png`, `37_metrics_vs_threshold.png`.

## 2. Overfitting check (bias vs variance)

| Model | Train AUC | CV AUC | Holdout AUC | Train − CV | Holdout − CV |
|---|---|---|---|---|---|
| Logistic regression | 0.7642 | 0.7600 | 0.7641 | +0.0042 | +0.0041 |
| Decision tree | 0.7388 | 0.7266 | 0.7269 | +0.0122 | +0.0003 |
| Random forest | 0.9353 | 0.7592 | 0.7618 | +0.1762 | +0.0026 |
| XGBoost | 0.8628 | 0.7698 | 0.7744 | +0.0930 | +0.0046 |
| LightGBM | 0.8433 | 0.7758 | 0.7819 | +0.0675 | +0.0061 |

Logistic regression and the depth-6 tree hardly overfit but rank worse than the boosting models (high bias); in the
tree grid, depth 12 with leaves of 50 reaches a training AUC of 0.8295 but a CV AUC of only 0.6792 (high variance).
The random forest's fully grown trees memorise the training data (train AUC 0.9353), yet averaging 400 of them keeps
CV AUC at 0.7592 (bagging reduces variance). The regularised boosting models have moderate gaps and the best CV and
holdout AUC. Holdout AUC is within +0.0003 to +0.0061 of CV for every model. Figure: `38_overfitting_check.png`.

## 3. K-Means clustering (notebook 05)

Eight features (age, log income, credit/income, annuity/income, years employed, `EXT_SOURCE_MEAN`, `INST_LATE_SHARE`,
`PREV_REFUSED_SHARE`); median imputation and scaling fitted on train. **k = 6**: the best silhouette for k ≥ 3 (0.1634
on a 20,000-applicant sample; k = 2 scores 0.1755 but its two halves default at 7.25% and 8.45%) and the elbow of the
inertia curve (drop of 107,958 to k = 6, 76,238 to k = 7).

| Segment | Applicants (train) | Share | Default rate (train) | Default rate (holdout) | Mean age | Mean EXT_SOURCE_MEAN | Mean INST_LATE_SHARE | Mean PREV_REFUSED_SHARE |
|---|---|---|---|---|---|---|---|---|
| Long-tenure employees (low risk) | 20,938 | 8.5% | 3.81% | 4.10% | 48.3 | 0.569 | 0.061 | 0.076 |
| Older applicants with strong external scores (low risk) | 66,471 | 27.0% | 4.00% | 4.11% | 53.8 | 0.597 | 0.045 | 0.048 |
| High loan-to-income borrowers (medium risk) | 38,227 | 15.5% | 7.23% | 6.66% | 46.5 | 0.525 | 0.058 | 0.060 |
| Young, early-career applicants (high risk) | 69,644 | 28.3% | 10.07% | 10.19% | 33.4 | 0.435 | 0.049 | 0.034 |
| Frequent late payers (high risk) | 18,330 | 7.5% | 12.70% | 12.86% | 42.5 | 0.476 | 0.461 | 0.112 |
| Often refused before (high risk) | 32,395 | 13.2% | 13.27% | 13.12% | 41.3 | 0.450 | 0.073 | 0.461 |

Names come from each cluster's most extreme standardised centre; the risk level compares the training default rate
with the overall 8.07% (low below 0.8×, high above 1.2×). High loan-to-income borrowers (mean credit/income 8.244 vs at
most 3.816 elsewhere) default below average. Adding the segment label to LightGBM changes CV AUC from 0.7758 to 0.7763
and holdout AUC from 0.7819 to 0.7813 (−0.0006): no gain. Figures: `39_kmeans_elbow_silhouette.png`,
`40_kmeans_cluster_heatmap.png`, `41_kmeans_default_rate.png`, `42_kmeans_clusters_2d.png`.

## 4. Cost-optimal threshold and business impact (LightGBM, notebook 04)

Assumed costs: approving a defaulter = 0.60 × AMT_CREDIT (LGD), rejecting a good customer = 0.15 × AMT_CREDIT (lost
margin). The threshold was chosen on the training out-of-fold predictions: **0.70** (the threshold that would minimise
holdout cost, computed for reference only, is also 0.70).

| Policy (holdout, 61,502 applicants) | Approval rate | Default rate among approved | Defaulters approved | Good applicants rejected | Total cost | Saving vs approve all |
|---|---|---|---|---|---|---|
| Approve everyone | 100.00% | 8.07% | 4,965 | 0 | 1,665,214,629 | — |
| Threshold 0.50 | 71.34% | 3.70% | 1,622 | 14,285 | 1,756,164,150 | −90,949,520 (−5.46%) |
| **Threshold 0.70 (cost-optimal)** | **89.44%** | **5.60%** | 3,080 | 4,608 | **1,431,358,554** | **233,856,075 (14.04%)** |

The optimal threshold costs 18.50% less than 0.5. Sensitivity (LGD 0.4–0.8 × MARGIN 0.10–0.20, all assumptions):
the threshold ranges from 0.57 to 0.83, approval from 78.52% to 97.35%, saving vs approving everyone from 4.53% to
30.04%. Figures: `21_cost_vs_threshold.png`, `22_approval_vs_default_rate.png`, `23_threshold_sensitivity.png`.

## 5. Thin-file findings

| Holdout group | Applicants | Default rate | LightGBM AUC (95% bootstrap interval) | Logistic regression AUC |
|---|---|---|---|---|
| All | 61,502 | 8.07% | 0.7819 (0.7756–0.7880) | 0.7641 |
| Bureau history: yes | 52,747 | 7.73% | 0.7837 (0.7764–0.7914) | 0.7678 |
| **Bureau history: no** | **8,755** | **10.17%** | **0.7645 (0.7512–0.7794)** | 0.7348 |
| Previous application: yes | 58,211 | 8.20% | 0.7811 (0.7755–0.7871) | 0.7633 |
| Previous application: no | 3,291 | 5.74% | 0.7882 (0.7602–0.8227) | 0.7655 |

| LightGBM ablation (same tuned parameters) | Features | Holdout AUC, all | Holdout AUC, no bureau history |
|---|---|---|---|
| Full model | 161 | 0.7819 | 0.7645 |
| Without EXT_SOURCE scores | 156 | 0.7548 | 0.7499 |
| Without bureau features | 150 | 0.7783 | 0.7654 |
| **Alternative data only** (no EXT_SOURCE, no bureau) | 145 | 0.7373 | **0.7507 (0.7341–0.7677)** |

Alternative data (application form + Home Credit's own previous-application and repayment history) alone ranks
bureau-less applicants with AUC 0.7507, only 0.0138 below the full model on that group, and better than it ranks
applicants who have a bureau file (0.7331). Among the 44,020 applicants without bureau history, 94.39% have a previous
Home Credit application. Figures: `12_thin_file_groups.png`, `24_auc_by_history_group.png`, `25_ablation_auc.png`,
`31_shap_beeswarm_no_bureau.png`.

## 6. Top SHAP drivers (LightGBM, 5,000 holdout applicants, mean |SHAP| in log-odds)

| Rank | Feature | Mean \|SHAP\| | Direction |
|---|---|---|---|
| 1 | EXT_SOURCE_MEAN | 0.3610 | lower external score → higher risk |
| 2 | CREDIT_TERM (annuity / credit) | 0.1265 | non-monotonic: mid-range values raise risk (D5 +0.193); the highest decile (annuity 9.35–12.12% of credit, the shortest loans) lowers it most (−0.333) |
| 3 | PREV_APP_CREDIT_RATIO_MEAN | 0.1208 | lower → higher risk |
| 4 | GOODS_CREDIT_RATIO | 0.1160 | lower (loan exceeds goods price) → higher risk |
| 5 | ORGANIZATION_TYPE | 0.1124 | categorical |
| 6 | CODE_GENDER | 0.1095 | male → higher predicted risk |
| 7 | EXT_SOURCE_3 | 0.1046 | lower → higher risk |
| 8 | INST_LATE_SHARE | 0.1007 | more late instalments → higher risk |

For bureau-less applicants under the alternative-data-only model the top drivers are CREDIT_TERM (0.1549),
CODE_GENDER (0.1418), PREV_APP_CREDIT_RATIO_MEAN (0.1388), GOODS_CREDIT_RATIO (0.1331) and INST_LATE_SHARE (0.1222).
Directions come from the mean-SHAP-by-value-decile table in notebook 04, section C (D1 = lowest value).
Figures: `26_shap_beeswarm.png`, `27_shap_bar.png`, `28`–`30_shap_waterfall_applicant_*.png` (reason codes in
notebook 04, section C2).

## 7. Fairness findings (holdout, threshold 0.70)

| Group | Applicants | Actual default rate | Approval rate | Good applicants rejected |
|---|---|---|---|---|
| Female | 40,544 | 6.99% | 91.95% | 6.25% |
| Male | 20,958 | 10.16% | 84.60% | 11.97% |
| Age 20–29 | 8,917 | 11.24% | 79.30% | 17.06% |
| Age 60–69 | 7,215 | 4.99% | 97.23% | 2.14% |

Dropping CODE_GENDER: holdout AUC 0.7819 → 0.7803, holdout cost +13,498,389 (0.94%), approval gap (F − M) 7.35 →
5.10 percentage points. The gap persists because men's actual default rate is higher and because gender can be
predicted from the other features with AUC 0.9094 (mainly OCCUPATION_TYPE). Figure: `32_fairness_gender_age.png`.
