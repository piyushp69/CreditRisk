# Progress log

Project: *Predicting loan default for thin-file borrowers using alternative data* (Home Credit Default Risk, Kaggle).

## Phase 0 — Data setup (2026-10-27)

- Downloaded 6 competition files with the Kaggle CLI into `data/raw/`, converted to downcast Parquet in
  `data/processed/` (`data/make_parquet.py`). See `data/README.md`.

## Phase 1 — Data understanding, cleaning, EDA (2026-10-27)

**Done**
- `notebooks/01_eda.ipynb` (runs end to end, ~12 s).
- `src/config.py` (paths, `RANDOM_STATE = 42`), `src/plotting.py` (chart style, `savefig` at dpi=150, default-rate
  bar chart), `src/eda.py` (missing-value and default-rate tables), `src/cleaning.py` (`clean_application` and
  report helpers), `src/features.py` (ratios and history flags used in EDA; side-table aggregates for Phase 2).
- Figures `01_target_distribution.png` … `12_thin_file_groups.png` in `reports/figures/`.

**Key numbers**
- `application_train`: 307,511 × 122; TARGET = 1 for 24,825 (8.07%), 11.39 repaid per default.
- 67 columns have missing values; 41 are >50% missing (39 building-information columns).
- `DAYS_EMPLOYED = 365243`: 55,374 rows (18.01%; 55,352 pensioners), default 5.40% vs 8.66% → NaN + flag.
- `CODE_GENDER = 'XNA'`: 4 rows dropped → 307,507 rows after cleaning.
- `AMT_INCOME_TOTAL`: 99.9th percentile 900,000, max 117,000,000; 278 rows above the 99.9th percentile (kept).
- Default rate by age: 12.31% (20–24) → 3.72% (65–69).
- EXT_SOURCE_1/2/3 correlation with TARGET: −0.1553 / −0.1605 / −0.1789 (strongest numeric signals).
- No bureau history: 44,020 applicants (14.32%), default 10.12% vs 7.73% with history.
- No previous application: 16,454 (5.35%), default 5.96% vs 8.19%.

**Issues / notes**
- pandas 3.0 reads text columns as `str` (not `object`); code handles both.
- EDA uses all labelled applicants (before the Phase 3 holdout split). Nothing is fitted here; the only
  downstream use is the choice of features, which is a common and minor form of researcher-level look-ahead.
- First render had annotation labels overlapping the overall-rate reference line; fixed in `plot_rate_bars`.

## Phase 2 — Feature engineering (2026-10-28)

**Done**
- `notebooks/02_feature_engineering.ipynb` (runs end to end, ~21 s); logic in `src/features.py`.
- `data/processed/model_table.parquet`: 307,507 rows × 163 columns (161 features + `SK_ID_CURR` + `TARGET`),
  139.9 MB in memory, 45.0 MB on disk, float32 for all non-integer numerics.
- Figures `13_default_rate_by_inst_late_share.png`, `14_default_rate_by_prev_refused_share.png`,
  `15_default_rate_by_ext_source_mean_decile.png`.

**Key numbers**
- Coverage: no bureau history 14.32%, no previous application 5.35%, no instalment history 5.16%.
- Of the 44,020 applicants without bureau history, 94.39% have a previous application and 94.59% have instalment history.
- `INST_LATE_SHARE`: default 6.80% (never late) → 15.43% (>50% of instalments late).
- `PREV_REFUSED_SHARE`: default 7.07% (none refused) → 15.93% (>50% refused).
- `EXT_SOURCE_MEAN` deciles: 22.93% (lowest) → 2.04% (highest); r = −0.2221 with TARGET.

**Issues / decisions**
- Instalment payments are consolidated per instalment before computing lateness/underpayment: 8.90% of payment
  rows are below the instalment amount, but only 0.0426% of instalments are underpaid once partial payments are
  summed. `INST_UNDERPAID_SHARE` is therefore almost constant; added `INST_PARTIAL_PAYMENT_SHARE` (not in the
  original spec) to keep the partial-payment signal.
- Count features (`*_COUNT`) are 0 for applicants without side-table rows; other aggregates stay NaN.
- `HAS_INSTALLMENT_HISTORY` added alongside the two requested history flags.
- 336,768 previous applications have `AMT_CREDIT = 0`; their application/credit ratio is NaN (safe division).

## Phase 3 — Modelling (2026-10-28)

**Done**
- `notebooks/03_modeling.ipynb` (runs end to end, ~14 min; the LightGBM search is ~11 min of that); logic in
  `src/modeling.py`.
- Stratified 80/20 split saved once to `data/processed/split_ids.parquet`: 246,005 train (8.0730% default) /
  61,502 holdout (8.0729%). Later runs reload it.
- Models in `models/` (`logreg_pipeline.joblib`, `lgbm_model.joblib`, `xgb_model.joblib`, `lgbm_best_params.json`);
  `reports/metrics_table.csv`, `reports/lgbm_search_results.csv`; `data/processed/oof_lgbm.parquet` (train
  out-of-fold predictions) and `data/processed/holdout_predictions.parquet`.
- Figures `16_roc_curves.png` … `20_confusion_matrices.png`.

**Key numbers**
| Model | CV AUC (train) | Holdout AUC | Gini | PR-AUC | KS |
|---|---|---|---|---|---|
| Logistic regression | 0.7600 ± 0.0013 | 0.7641 | 0.5281 | 0.2482 | 0.3989 |
| LightGBM | 0.7758 ± 0.0015 | 0.7819 | 0.5638 | 0.2818 | 0.4237 |
| XGBoost (untuned) | 0.7698 ± 0.0017 | 0.7744 | 0.5489 | 0.2693 | 0.4079 |
- Best LightGBM (20-trial random search): num_leaves 15, learning_rate 0.02, min_child_samples 100,
  colsample_bytree 0.3, subsample 0.8, reg_lambda 10; final model 2,203 trees.
- Leakage guard: max holdout AUC 0.7819 (< 0.85). Accuracy of "approve everyone" on holdout: 91.93%.
- Scores are not calibrated (LightGBM mean prediction 0.3737 vs default rate 0.0807) because of class weighting.

**Issues / decisions**
- First LightGBM attempt stopped after 1 tree: with `scale_pos_weight`, the default binary_logloss on the unweighted
  early-stopping set worsens immediately and triggered early stopping. Fixed by `metric="auc"` only.
- LightGBM 4.7 deprecates `eval_set`; using `eval_X` / `eval_y`.
- Early stopping uses a stratified 10% carve-out of the *fitting* data, inside every CV fold and for the final model,
  so scored folds and the holdout are never used for stopping.
- XGBoost uses fixed parameters (optional comparison model, not tuned).
- Re-running the notebook reproduces `metrics_table.csv` and all holdout predictions exactly (checked).

## Phase 4 — Threshold, thin-file analysis, SHAP, fairness (2026-10-28)

**Done**
- `notebooks/04_threshold_explainability.ipynb` (runs end to end, ~3.5 min); logic in `src/cost.py`,
  `src/explain.py`, `src/modeling.py`.
- Figures `21_cost_vs_threshold.png` … `32_fairness_gender_age.png`.

**Key numbers**
- Cost-optimal threshold (LGD 0.60, MARGIN 0.15) chosen on train OOF: 0.70 (holdout optimum, for reference: 0.70).
- Holdout cost: approve everyone 1,665,214,629; threshold 0.50 1,756,164,150 (−5.46%); threshold 0.70
  1,431,358,554 (saving 233,856,075 = 14.04%; approval 89.44%; default rate among approved 5.60%).
- Sensitivity grid: thresholds 0.57–0.83, approval 78.52%–97.35%, saving 4.53%–30.04%.
- Holdout AUC (LightGBM): no bureau history 0.7645 [0.7512–0.7794] vs with bureau 0.7837; no previous application
  0.7882 [0.7602–0.8227].
- Ablation (holdout AUC all / no-bureau): full 0.7819 / 0.7645; no EXT_SOURCE 0.7548 / 0.7499; no bureau 0.7783 /
  0.7654; alternative data only 0.7373 / 0.7507.
- Top SHAP drivers: EXT_SOURCE_MEAN 0.3610, CREDIT_TERM 0.1265, PREV_APP_CREDIT_RATIO_MEAN 0.1208,
  GOODS_CREDIT_RATIO 0.1160, ORGANIZATION_TYPE 0.1124, CODE_GENDER 0.1095.
- Fairness at 0.70: approval F 91.95% vs M 84.60% (actual default 6.99% vs 10.16%); good applicants rejected
  6.25% vs 11.97%. Without CODE_GENDER: AUC 0.7803, gap 7.35 → 5.10 pp, cost +13,498,389; gender predictable from
  other features with AUC 0.9094.

**Issues / decisions**
- The ablation models reuse the tuned LightGBM parameters (no re-tuning per feature set).
- "No bureau" ablation also drops `HAS_BUREAU_HISTORY`.
- Bootstrap intervals (200 resamples) added for group AUCs, which the spec did not require.
- SHAP values are in log-odds of the class-weighted model; additivity check max error 1.51e-14.
- Figure fixes after visual review: legend placement (cost plot, fairness panels), waterfall axis labels.

## Phase 5 — Deliverables (2026-10-28)

**Done**
- `reports/results_summary.md` (one page: metrics, threshold and business impact, thin-file results, SHAP drivers,
  fairness), `reports/report_draft.md` (draft academic report with `[ADD YOUR INSIGHT]` markers),
  `requirements.txt` (versions actually used), root `README.md` (structure and run order).
- Final check: all four notebooks re-run end to end in order (13 s, 21 s, 841 s, 195 s) with no errors, no stderr
  output and no unfilled insight cells.

**Open items for the author**
- Fill the `[ADD YOUR INSIGHT]` markers and verify the reference list against the required citation style.
- The external financial-inclusion statistics in the introduction need a cited source (none were invented).

## Final verification against the master plan (2026-10-28)

**Checked**
- Every numbered step of Phases 1–5 has its files and outputs: 4 notebooks, 32 figures, model table (307,507 × 163),
  split ids, 3 models + best parameters, metrics table, OOF/holdout predictions, results summary, report draft,
  requirements, README. All notebooks are fully executed, execution counts are sequential, and there are no error or
  stderr outputs. Each figure cell is followed by an insight cell.
- Every number in `results_summary.md` and `README.md` appears verbatim in a notebook output. The remaining numbers
  in the report draft and insight cells are derived (sums, differences, ratios) and were checked by hand.
- The SHAP "Direction" column was checked numerically: mean SHAP by decile of each top driver's value.

**Fixed**
- `CREDIT_TERM` direction was wrong ("higher → higher risk"). It is non-monotonic: rank correlation between value
  and SHAP is 0.073, mid-range deciles raise risk (D5 +0.193) and the highest decile (annuity 9.35–12.12% of the
  credit, the shortest loans) lowers it the most (−0.333). Added a direction-check cell to notebook 04 (section C,
  before the beeswarm), rewrote the beeswarm insight from its output and corrected `results_summary.md`.
- Notebook 01: annuity/income decile spread corrected from 1.71 to 1.72 percentage points (exact 1.7187; 1.71 came
  from subtracting rounded rates).
- Notebook 04 re-run end to end (209 s): every earlier output is unchanged; only the new cell was added.

## Phase 6 — Report completion and Streamlit app (2026-11-15)

**Done**
- Report finished and renamed `reports/report_draft.md` → `reports/report.md`. All eight `[ADD YOUR INSIGHT]`
  markers replaced: financial-inclusion statistics (Global Findex 2025: 79% of adults with an account in 2024,
  1.3 billion without; CFPB 2025 update: 2.7% of US adults credit invisible in 2020, a further 9.8% unscored),
  why first-time customers default less, whether AUC ≈ 0.78 is good enough (Kaggle leaderboard context), realistic
  LGD / margin values, implications for genuinely new-to-credit customers, fairness definitions and legal
  constraints (ECOA / Regulation B, Directive 2004/113/EC, EU AI Act), and the overall recommendation. Ten
  references added (sources checked online on 2026-11-15).
- New supporting cells, appended at the end of the notebooks so no earlier output changes:
  notebook 02 section 7 (first-time Home Credit customers), notebook 04 last section (applicants with neither bureau
  nor previous-application data). Both notebooks re-run end to end; every earlier output is identical (only pandas
  Styler's random HTML ids differ).
- Streamlit app in `app/` (9 pages) with a data bundle built by `app/build_assets.py` in `data/processed/app/`.
  Dynamic: one global decision policy (LGD, margin, cost-optimal or manual threshold) in the sidebar drives every
  page and is kept in the URL; filters, segment selectors, click-to-drill-down, row selection, what-if re-scoring,
  scenario buttons, CSV downloads. Responsive: top navigation that collapses into the sidebar on phones, auto-fit KPI
  and card grids (2 per row on phones), stacked columns, full-width Plotly charts, chart titles outside the plot so
  they wrap, light and dark palettes (validated steps for each).

**Key numbers (new)**
- First-time applicants: 16,454, of whom 84.99% have a bureau record; default 5.72% with a record vs 7.29% without
  (2,470); revolving loans 17.56% vs 9.07% for returning customers; default 5.96% (0 previous applications), 8.38%
  (1), 9.68% (10+); returning customers 7.07% (no refusal) → 15.93% (more than half refused).
- Holdout applicants with neither history: 480 (35 defaults, 7.29%); LightGBM AUC 0.7893 (95% bootstrap interval
  0.7295–0.8564), too wide for a conclusion.
- Unlabelled test applicants scored by the final model: 48,744, mean score 0.3663; at threshold 0.70, 90.35% would be
  approved; population stability index vs holdout scores 0.003.

**Checks**
- `app/build_assets.py` refits the three ablation models and the model without CODE_GENDER; it asserts their holdout
  AUCs reproduce notebook 04 (0.7548, 0.7783, 0.7373, 0.7803) and the no-gender OOF threshold is 0.72. SHAP values
  come from LightGBM's own TreeSHAP (`pred_contrib`); the 5,000-applicant sample reproduces mean |SHAP| 0.3610 for
  EXT_SOURCE_MEAN.
- The app's vectorised cost sweep equals `src.cost.threshold_sweep` exactly (max difference 0.0 on every column);
  the optimal threshold and the 3 × 3 sensitivity grid match; the bootstrap reproduces notebook 04's intervals
  (e.g. 0.7645 [0.7512–0.7794] without bureau history; 0.7507 [0.7341–0.7677] for alternative data only).
- Every page and 21 interaction paths run without exceptions under `streamlit.testing.AppTest`; pages were
  screenshotted at 1440 px and 390 px (no horizontal overflow) and in dark mode.

**Issues / decisions**
- Windows Application Control blocked `jupyter-nbconvert.exe` and, once, shap's C extension inside the kernel
  (shap printed "C extension was not built"; values were unchanged). Notebooks are run with `python -m nbconvert`
  and notebook 04 was re-run so its outputs are clean. The README now uses `python -m` commands.
- Row-level app data stays in the git-ignored `data/processed/app/`, consistent with the competition rules.
- Fairness in the app: the model without gender uses its own OOF-optimal threshold under the current costs, as in
  notebook 04.

**Open items for the author**
- The report's interpretive paragraphs were drafted for the author; read them, adjust them to your own view and
  wording, and check every reference against the department's citation style.

## Phase 7 — Stage 1: five-model comparison and K-Means clustering (2026-11-17)

**Change of plan (same day).** An earlier instruction (MLP, PCA, probability calibration, bootstrap intervals,
bias-variance notebook) was started and then withdrawn by the author. That work was stopped and removed: the
notebook 03 run that included it was aborted during its random-forest search; the MLP/calibration/bootstrap/PCA code
was deleted from `src/`; no notebook, model, figure or report from it was kept. The aborted run had only rewritten
`reports/lgbm_search_results.csv`, which was identical apart from timings and was restored from the backup.

**Done**
- `notebooks/03_modeling.ipynb` rebuilt around five models and run end to end (45 min, 21:24 finish; no errors, no
  stderr in any cell): logistic regression, LightGBM and XGBoost cells kept as before; new decision tree and random
  forest; cost-optimal threshold per model; one holdout evaluation; comparison charts; metric-vs-threshold curves;
  overfitting check. The LightGBM search reran with identical results, and the saved logistic-regression, LightGBM
  and XGBoost holdout predictions are identical to the previous run (max difference 0).
- `notebooks/05_clustering.ipynb` (new, ~2 min, no errors): K-Means on 8 features, k from elbow + silhouette, named
  segments, 2-D views, cluster label as a LightGBM feature.
- `src/`: `modeling.py` (shared tree preprocessing, `make_dt_pipeline`, `make_rf_pipeline`, `cross_validate_sklearn`
  with OOF predictions, `metric_row`, log loss), `thresholds.py` (new), `clustering.py` (new), `cost.py`
  (`cost_components`, `optimal_threshold_table`), `plotting.py` (`plot_confusion`), `explain.py` (shap imported
  lazily so reason codes work without shap).
- Outputs: `reports/metrics_table_final.csv` (one row per model per threshold type), `cv_results.csv`,
  `overfitting_check.csv`, `dt_search_results.csv`, `rf_search_results.csv`, `cluster_profiles.csv`;
  `data/processed/threshold_curves.parquet`, `oof_predictions.parquet`, `holdout_predictions.parquet` (now 5 models +
  `AMT_CREDIT`), cluster outputs; `models/dt_pipeline.joblib`, `rf_pipeline.joblib`, `kmeans_segments.joblib`,
  `*_params.json`, `cost_thresholds.json`. Figures 16, 17, 19, 20 updated to five models; new figures 33–42.
- `reports/results_summary.md` updated; `reports/report.md` restructured to the rubric order (cover page, abstract,
  sections 1–13, references; four `[ADD YOUR INSIGHT]` markers); README updated (notebook order 01→05).

**Key numbers (holdout, 61,502 applicants)**
| Model | CV AUC | Holdout AUC | Gini | PR-AUC | KS | Log loss | Acc@0.5 | F1@0.5 | Threshold | Acc@opt | Prec@opt | Rec@opt | F1@opt | Saving@opt | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic regression | 0.7600 ± 0.0013 | 0.7641 | 0.5281 | 0.2482 | 0.3989 | 0.5778 | 0.6997 | 0.2726 | 0.74 | 0.8832 | 0.2894 | 0.3072 | 0.2980 | 10.74% | 18.7 s |
| Decision tree | 0.7266 ± 0.0022 | 0.7269 | 0.4539 | 0.1988 | 0.3346 | 0.6032 | 0.6507 | 0.2410 | 0.76 | 0.8950 | 0.2825 | 0.1950 | 0.2307 | 6.72% | 16.8 s |
| Random forest | 0.7592 ± 0.0022 | 0.7618 | 0.5236 | 0.2495 | 0.3987 | 0.3601 | 0.8752 | 0.3021 | 0.52 | 0.8828 | 0.2877 | 0.3065 | 0.2968 | 10.82% | 99.8 s |
| XGBoost | 0.7698 ± 0.0017 | 0.7744 | 0.5489 | 0.2693 | 0.4079 | 0.5045 | 0.7556 | 0.2964 | 0.70 | 0.8805 | 0.2892 | 0.3293 | 0.3079 | 11.65% | 19.3 s |
| LightGBM | 0.7758 ± 0.0015 | 0.7819 | 0.5638 | 0.2818 | 0.4237 | 0.5194 | 0.7414 | 0.2959 | 0.70 | 0.8750 | 0.2903 | 0.3797 | 0.3290 | 14.04% | 17.9 s |
- Decision tree: `max_depth` 6, `min_samples_leaf` 200 (64 leaves); `EXT_SOURCE_MEAN` = 81.11% of its impurity decrease.
- Random forest: 400 trees, `max_depth` None, `min_samples_leaf` 20, `max_features` 0.3; 3-fold CV AUCs of the 12
  sampled configurations 0.7552–0.7584.
- Overfitting check (train − CV AUC): LR 0.0042, DT 0.0122, RF 0.1762, XGBoost 0.0930, LightGBM 0.0675; holdout − CV
  +0.0003 to +0.0061. Tree grid: depth 12 / leaf 50 → train 0.8295, CV 0.6792.
- At 0.5, four models cost more than approving everyone (LR −17.52%, DT −32.33%, XGBoost −3.91%, LightGBM −5.46%);
  random forest saves 10.61%. Approving everyone: 91.93% accuracy.
- K-Means: k = 6 (silhouette 0.1634; k = 2 scored 0.1755 but its halves default at 7.25% / 8.45%). Segments (train
  default rate): long-tenure employees 3.81%, older applicants with strong external scores 4.00%, high loan-to-income
  borrowers 7.23%, young early-career applicants 10.07%, frequent late payers 12.70%, often refused before 13.27%;
  holdout rates within 0.56 percentage points. Cluster label in LightGBM: CV AUC 0.7758 → 0.7763, holdout AUC
  0.7819 → 0.7813 (−0.0006).
- Leakage guard: max holdout AUC 0.7819 (< 0.85).

**Issues / decisions**
- Random-forest tuning is time-bounded: 12 of the 36 combinations were sampled (`ParameterSampler`, budget 35 min);
  all 12 finished (19.4 min). XGBoost keeps its original fixed parameters (not tuned), as before.
- Logistic-regression CV now uses `src.modeling.cross_validate_sklearn` (same folds) to obtain OOF predictions for
  its threshold; fold AUCs are unchanged.
- Training time = one final fit on the full training set; random forest, XGBoost and LightGBM use all CPU cores, the
  boosting models fit on 90% of train (10% early-stopping carve-out).
- `metrics_table_final.csv` is rounded to 6 decimals, which double-rounds one value (LR recall at 0.74 = 1,525/4,965 =
  0.307150… is stored as 0.307150 and formats as 0.3071; exact 0.3072). Reports and the website use metrics recomputed
  at full precision from `holdout_predictions.parquet` (max difference from the CSV 4.9e-7).
- `models/rf_pipeline.joblib` is 57.0 MB (compressed): too large for the website bundle and above GitHub's 50 MB
  warning size.
- K-Means: k = 2 is excluded by rule (best silhouette among k ≥ 3) because its two groups have near-equal default
  rates; cluster names come from each cluster's most extreme standardised centre (rule in the notebook), risk levels
  from the default rate vs 0.8× / 1.2× the overall rate. K-Means was fitted on all training applicants, so the CV AUC
  of LightGBM + cluster has a little unsupervised look-ahead; the holdout comparison is clean.
- Notebook 05 was re-run once after moving a chart legend; all outputs identical (apart from 16th-digit float noise).
- Notebook 04 was not re-run: its inputs (LightGBM model, OOF and holdout predictions) are unchanged.
- On Windows, joblib's background resource tracker prints `KeyError` tracebacks to the console after parallel runs;
  they are not in any notebook output and do not affect results.
- `reports/report_draft.md` was renamed `reports/report.md` in Phase 6; that file was restructured.

**Open items for the author**
- Fill the four `[ADD YOUR INSIGHT]` markers and the submission date in `reports/report.md`.

## Phase 8 — Stage 2: project website (2026-11-17)

**Done**
- New Streamlit website: `app/app.py` (st.navigation, sidebar sections, logo), 12 pages in `app/site_pages/`
  (Home, Dataset, Preprocessing, Exploratory analysis, Clustering, How each model works, Model comparison, Business
  decision, Thin-file & explainability, Fairness, Live risk predictor, Conclusion & future scope), helpers in
  `app/utils/` (cached loaders, model colours, Plotly charts, page blocks, model texts, predictor), custom light/dark
  theme in `.streamlit/config.toml`, `app/requirements.txt` (pinned, website only), `DEPLOY.md`.
- `app/build_assets.py` (~3 min) writes `app/assets/` (3.8 MB: CSV / JSON / Parquet aggregates, ROC/PR points
  downsampled to ≤500 per model, threshold curves, threshold grid, EDA aggregates, cluster outputs, thin-file, SHAP and
  fairness tables, holdout predictions without `SK_ID_CURR`, two figures) and `app/models/lgbm_model.txt` (3.8 MB,
  LightGBM booster at its best iteration). Checks built in: metrics recomputed from the holdout predictions match
  `reports/metrics_table_final.csv` (relative difference < 1e-5); thin-file AUCs and bootstrap intervals, ablation AUCs,
  SHAP top driver (EXT_SOURCE_MEAN 0.3610), the no-gender model (AUC 0.7803, threshold 0.72) and the proxy AUC
  (0.9094) reproduce notebook 04; each model's OOF threshold grid reproduces `models/cost_thresholds.json` at the base
  costs; the booster file reproduces the saved LightGBM's holdout scores; the three predictor examples (low / medium
  / high risk, real holdout applicants) reproduce their holdout scores exactly through the form pipeline
  (0.110 / 0.488 / 0.823).
- The old Phase-6 app (`Home.py`, `dashboard.py`, `core/`, `views/`, `dashboard_pages/`, its `build_assets.py`) was
  moved unchanged to `legacy_app/` (imports updated; starts without exceptions).
- README updated; `.gitignore` now also excludes `models/rf_pipeline.joblib` (57 MB) and local `_*.log` files.

**Checks**
- `streamlit.testing.AppTest`: all 12 pages run with no exceptions and no Python warnings; interactions tested (each
  example button, editing inputs, threshold slider 0.05–0.95, every metric/threshold selector, every model × three
  LGD/margin pairs on the business page, manual threshold, every EDA dropdown value, every toggle on the models,
  clustering and fairness pages). Business page at the base costs shows thresholds 0.74 / 0.76 / 0.52 / 0.70 / 0.70 and
  savings 10.74% / 6.72% / 10.82% / 11.65% / 14.04% (LR, DT, RF, XGBoost, LightGBM), and −5.46% for LightGBM at 0.5,
  as in `metrics_table_final.csv`.
- Browser (Chrome, local server): every page opened; dark and light themes; no console errors; at 390 px width (pages
  loaded in 390 px iframes) no page scrolls horizontally and every chart renders.
- No result is typed into page code: all numbers come from `app/assets/` (literal numbers left in the code are UI
  settings such as slider ranges and axis limits, or method constants).

**Issues / decisions**
- Pages live in `app/site_pages/`, not `app/pages/`: with a folder named `pages/` next to the entry script, Streamlit
  1.63 still serves a deep link (e.g. `/comparison`) on a fresh server through its legacy auto-navigation, skipping
  `app.py` (no layout, logo or sections). Found in the browser check; renaming fixed it.
- The website does not use the `shap` package (its numba DLL was blocked by Windows Application Control at the start
  of this session); SHAP values come from LightGBM's own TreeSHAP (`pred_contrib`). `src/explain.py` now imports shap
  lazily so its reason-code helpers work without it.
- Waterfall PNGs 28–30 were not shipped because their titles contain `SK_ID_CURR`; the website rebuilds the three
  example waterfalls in Plotly from SHAP values without identifiers.
- The live predictor fills every feature not on the form from the starting applicant (training medians / modes, or a
  loaded example). Changing the average external score shifts the available external scores by the same amount;
  changing refused applications updates the previous-application counts and shares.
- Business page thresholds are looked up from a grid chosen on the training OOF predictions (LGD 0.20–0.90 step 0.05,
  margin 0.05–0.30 step 0.01); the holdout curve is recomputed live but never used to choose.
- Confusion-matrix and reference-line labels were shortened / moved after the visual check (they overlapped at
  narrow widths); the predictor gauge got wider margins.
- Not verified: deployment on Streamlit Community Cloud itself (Python version availability for the pinned packages).

**Open items for the author**
- Add the GitHub and deployed-site links (README, Conclusion page "About" box) after publishing (`DEPLOY.md`).

## Phase 9 — Report insights (2026-11-17)

- The four `[ADD YOUR INSIGHT]` markers in `reports/report.md` (sections 5, 8, 9, 12) were replaced by drafted
  paragraphs wrapped in `<!-- INSIGHT START -->` / `<!-- INSIGHT END -->` for the author's review; submission date set
  to 17 November 2026. Every number in them comes from `reports/metrics_table_final.csv`, `reports/cluster_profiles.csv`,
  `reports/rf_search_results.csv`, `reports/lgbm_search_results.csv`, `reports/overfitting_check.csv` or notebook 03/04
  outputs; combined figures were computed from those files (low-risk segments 35.53% of applicants at 3.95% default,
  behavioural segments 20.62% at 13.07%; logistic-regression vs LightGBM holdout cost difference 54,982,702).

## Second cleanup (2026-11-17)

- Moved to `_to_delete/` (same relative paths, nothing permanently deleted): `legacy_app/` (earlier Phase-6 app,
  29 files), `data/processed/app/` (its row-level bundle, 6 files, 50.2 MB), `models/rf_pipeline.joblib` (57.0 MB),
  `models/xgb_model.joblib`, `models/kmeans_segments.joblib`, `reports/metrics_table.csv` (superseded by
  `metrics_table_final.csv`), `app/assets/score_histograms.csv` and `app/assets/cluster_plot_axes.csv` (no page read
  them). Nothing in the website, the report or the notebooks reads these files.
- `app/build_assets.py` no longer writes the two unused asset files. The notebook save lines stay: re-running notebook
  03 or 05 recreates the three model files and `metrics_table.csv`.
- README (folder tree, how-to-run, reproducibility), DEPLOY.md and `.gitignore` (now also excludes `_to_delete/`)
  updated. `__pycache__` folders deleted.
- Checks: all 42 figures present and every figure named in `report.md` / `results_summary.md` exists; all 10 `src/`
  modules import; `python app/build_assets.py` runs (169 s; all reproduction asserts pass; 37 of 38 outputs
  byte-identical, `facts.json` differs because it stores the build time); AppTest of all 12 pages: 0 exceptions;
  interaction tests pass; headless server healthy, all page URLs return 200.
- Size: 1947.3 MB before; 1836.5 MB without `_to_delete/` (110.6 MB, delete it by hand after review). Sizes are
  in MB of 1024² bytes.

## Phase 10 — IEEE conference paper (2026-11-18)

- `paper/`: `main.tex` (IEEEtran conference, 7 figures incl. a TikZ workflow, 5 tables, 7 equations), `references.bib`
  (25 references, each checked against Crossref/DataCite or the publisher page), `figures/` (6 PNGs copied from
  `reports/figures/`), `README.txt` (Overleaf instructions, alternative titles, verification source per reference).
  `paper_overleaf.zip` at the project root for upload to Overleaf.
- No TeX engine on this machine: the paper was not compiled; static checks pass (braces, environments, refs/labels,
  citations, table columns, figure files). Page count estimated at about 7.5 pages; confirm on Overleaf.
- Every number in `main.tex` was matched by script against the result CSVs and notebook outputs. Rounding slips fixed in
  the paper (not in the report): LightGBM − LR AUC gain 0.0179 (report.md says 0.0178); largest segment train/holdout
  default-rate difference 0.56 pp (report.md and notebook 05 say 0.57); Elkan (2001) pages 973–977 (official IJCAI).
- Author email left as `[ADD EMAIL]`.

## Phase 10b — Paper corrections and figure re-export (2026-11-18)

- Rounding slips corrected at the source: `reports/report.md` (AUC gain over logistic regression +0.0179, segment
  train/holdout gap 0.56 pp, Elkan pages 973–977), notebook 05 (two markdown cells: 0.56 pp; outputs unchanged) and the
  K-Means entry above. The Streamlit pages compute these values from `app/assets/` and already showed 0.56; all 12
  pages render without exceptions. Every number in `report.md` and `results_summary.md` was re-matched by script
  against the result CSVs and notebook code outputs.
- Paper figures re-exported by `paper/make_figures.py` for one IEEE column: 3.5 in wide, 300 dpi, all text ≥ 8 pt
  (checked in code); Figs. 3–5 are now single-column two-panel figures. Old full-width copies removed from
  `paper/figures/` (originals stay in `reports/figures/`).
- External facts verified: CPU name read from the system (13th Gen Intel(R) Core(TM) i7-13650HX; 24 GB RAM);
  Bussmann et al. 15,000 companies confirmed in the publisher's abstract deposited with the DOI; decision-tree grid
  copied from notebook 03; CFPB 2.7% / 7.0 million / 9.8% confirmed in the CFPB PDF itself (text and Table 2);
  Findex 79% and 1.3 billion confirmed on World Bank pages.
- Not compiled: running a downloaded TeX installer (TinyTeX, digest-verified, in the session scratchpad) was blocked
  by the tool's safety check; compile on Overleaf or locally once a TeX engine is approved.

## Phase 10c — Final paper edits (2026-11-18)

- `paper/main.tex`: KS notation G_1/G_0; abstract shortened to 235 words (all 18 numbers kept); `\fitwidth` guard
  (scales with `\resizebox` only if wider than the target width) around Fig. 1, Table III and the thin-file table;
  new Table IV, LightGBM holdout confusion matrix at t* = 0.70 (recomputed from `holdout_predictions.parquet`, equal to
  `metrics_table_final.csv`); unnumbered "Data and Code Availability" and "Acknowledgment" (AI-use disclosure
  placeholder) sections. Static checks pass; `paper_overleaf.zip` rebuilt. Still not compiled.
