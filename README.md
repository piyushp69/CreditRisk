# Predicting loan default for thin-file borrowers using alternative data

College predictive-analytics project on the official Kaggle competition data
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (provider: Home Credit Group).
It asks whether application, previous-application and repayment data can predict default for applicants with little
or no credit history, which decision threshold minimises a lender's cost, what drives rejections, and whether
decisions are fair across gender and age.

Five models are compared under one protocol (logistic regression, decision tree, random forest, XGBoost, LightGBM),
and K-Means clustering segments the applicants. Headline results (holdout, 61,502 applicants): LightGBM is best with
AUC 0.7819 (XGBoost 0.7744, logistic regression 0.7641, random forest 0.7618, decision tree 0.7269) and the highest F1
at its cost-optimal threshold (0.3290); that threshold (0.70) saves 14.04% vs approving everyone under the assumed
costs; six K-Means segments default at 3.81%–13.27%; AUC 0.7645 for applicants without Credit Bureau history. See
`reports/results_summary.md` and the full report `reports/report.md`.

A multi-page **Streamlit website** (`app/`) presents the whole project for a teacher, examiner or recruiter: the data,
preprocessing, EDA, clustering, how each model works, a detailed model comparison (every metric, a live threshold
slider, overfitting check, verdict), the business decision with adjustable costs, thin-file results and SHAP
explanations, fairness, a live loan-risk predictor and the conclusions. It ships no raw data (only aggregates and model
outputs in `app/assets/`), works on laptop and phone screens, in light and dark mode. Deployment: see `DEPLOY.md`.

## Folder structure

```
.
├── data/
│   ├── raw/                  original Kaggle CSVs (not in git)
│   ├── processed/            Parquet copies, model_table, split ids, predictions (not in git)
│   ├── make_parquet.py       CSV → downcast Parquet
│   └── README.md             data source, files used/excluded, download date
├── notebooks/
│   ├── 01_eda.ipynb                          data understanding, cleaning, EDA
│   ├── 02_feature_engineering.ipynb          builds data/processed/model_table.parquet
│   ├── 03_modeling.ipynb                     split, 5-model comparison (LR, decision tree, random forest, XGBoost,
│   │                                         LightGBM), cost-optimal thresholds, holdout evaluation, overfitting check
│   ├── 04_threshold_explainability.ipynb     cost threshold, thin-file analysis, SHAP, fairness (LightGBM)
│   └── 05_clustering.ipynb                   K-Means applicant segments, cluster feature in LightGBM
├── src/                      reusable code imported by the notebooks
│   ├── config.py             paths, RANDOM_STATE = 42
│   ├── cleaning.py           clean_application + cleaning reports
│   ├── eda.py                missing-value and default-rate tables
│   ├── features.py           ratios, BUR_/PREV_/INST_ aggregates, model table
│   ├── modeling.py           split, pipelines (LR, decision tree, random forest), boosting with early stopping,
│   │                         cross-validation with out-of-fold predictions, metrics
│   ├── thresholds.py         accuracy / precision / recall / F1 across thresholds (numpy only)
│   ├── clustering.py         K-Means features, scaler and model
│   ├── cost.py               cost model, threshold sweep, sensitivity grid, threshold lookup for many cost scenarios
│   ├── explain.py            SHAP helpers, plain-English reason codes
│   └── plotting.py           chart style, savefig (PNG, dpi 150)
├── app/                      project website (Streamlit)
│   ├── app.py                entry point: page config, sidebar navigation (st.navigation), logo
│   ├── site_pages/           12 pages: 01_home … 12_conclusion (not named pages/: see app.py)
│   ├── utils/                loaders (cached), style (model colours, formats), charts (Plotly), ui (page blocks),
│   │                         content (model texts), predictor (form → features → score → SHAP reasons)
│   ├── assets/               small CSV / JSON / Parquet bundle + two figures (built by build_assets.py, ~4 MB)
│   ├── models/lgbm_model.txt final LightGBM as a text model (~4 MB)
│   ├── build_assets.py       builds assets/ and models/ from the project outputs (run once, ~3 min)
│   ├── feature_dictionary_source.csv  formulas and meanings of the engineered features
│   └── requirements.txt      pinned packages for the website only
├── .streamlit/config.toml    website theme (light + dark) and server settings
├── DEPLOY.md                 GitHub + Streamlit Community Cloud deployment
├── paper/                    IEEE conference paper (main.tex, references.bib, figures/; see paper/README.txt)
├── models/                   trained models (joblib) + tuned parameters and cost-optimal thresholds (json)
├── reports/
│   ├── figures/              all plots, numbered 01_… to 42_…
│   ├── metrics_table_final.csv  holdout metrics, one row per model per threshold type (0.5 / cost-optimal)
│   ├── cv_results.csv, overfitting_check.csv, cluster_profiles.csv
│   ├── lgbm/dt/rf_search_results.csv
│   ├── results_summary.md    one-page results
│   └── report.md             academic report
├── PROGRESS.md               phase-by-phase log with key numbers and issues
├── requirements.txt          library versions actually used (Python 3.14.3)
└── .gitignore                excludes data/raw, data/processed, kaggle.json, the 57 MB random-forest model and _to_delete/
```

## How to run

1. **Environment:** Python 3.14 and `pip install -r requirements.txt`.
2. **Data:** accept the competition rules on Kaggle, place your API token in `~/.kaggle/kaggle.json`, then download
   the six files listed in `data/README.md` into `data/raw/` and unzip them.
3. **Parquet conversion:** `python data/make_parquet.py`
4. **Notebooks, in this order** (each depends on outputs of the previous one):

   | Order | Notebook | Produces | Runtime on the dev machine |
   |---|---|---|---|
   | 1 | `01_eda.ipynb` | figures 01–12 | ~15 s |
   | 2 | `02_feature_engineering.ipynb` | `data/processed/model_table.parquet`, figures 13–15 | ~20 s |
   | 3 | `03_modeling.ipynb` | `split_ids.parquet`, 5 models, `metrics_table_final.csv`, OOF/holdout predictions, `threshold_curves.parquet`, figures 16–20, 33–38 | ~45 min |
   | 4 | `04_threshold_explainability.ipynb` | figures 21–32 | ~3.5 min |
   | 5 | `05_clustering.ipynb` | cluster profiles and plotting sample in `data/processed/`, figures 39–42 | ~2 min |

   Run them in Jupyter, or headless from the project root:

   ```bash
   python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
   ```

5. **Website data bundle** (once, after the notebooks; ~3 min): `python app/build_assets.py`. It writes small
   aggregates and model outputs (no raw data, no `SK_ID_CURR`) to `app/assets/` and the LightGBM model to
   `app/models/lgbm_model.txt`. Metrics are recomputed from the saved predictions and checked against
   `reports/metrics_table_final.csv`; the thin-file, ablation, SHAP and fairness results of notebook 04 are recomputed
   with the same seeds and asserted to match; the three example applicants of the live predictor are checked to
   reproduce their real holdout scores exactly.
6. **Run the website:**

   ```bash
   python -m streamlit run app/app.py
   ```

   Then open http://localhost:8501. On a phone the page menu opens from the » button. To publish it for free, follow
   `DEPLOY.md`.

On Windows machines with Application Control / Smart App Control, the `jupyter-nbconvert.exe` and `streamlit.exe`
launchers can be blocked; the `python -m …` forms above avoid them.

## The website at a glance

| Page | What it shows |
|---|---|
| Home | Problem, research questions, headline cards (best model, its AUC and F1, cost saving, approval rate), pipeline diagram, guide |
| Dataset | Source and authenticity, files used / excluded, class imbalance, searchable column dictionary, data limitations |
| Preprocessing | Missing values and strategy per model, the DAYS_EMPLOYED anomaly, encoding, scaling, split and leakage diagram, feature formulas |
| Exploratory analysis | Default rate by any categorical variable (dropdown), age, external scores, affordability, correlations, thin-file groups, behaviour, numeric explorer, key findings |
| Clustering | Elbow and silhouette, one card per segment, default rate by segment, centre heatmap, 2-D views, segment label in LightGBM |
| How each model works | One tab per model: explanation and diagram, why chosen, preprocessing, settings and tuning, metrics, confusion matrix, ROC, strengths / weaknesses, bias / variance; tree plot for the decision tree |
| Model comparison | Full table (0.5 / cost-optimal) with winners highlighted, metric explorer, grouped bars, heatmap, radar, ROC / PR, confusion matrices, live threshold slider, overfitting check, everything-in-one-place table, verdict |
| Business decision | LGD and margin sliders, model selector, cost-vs-threshold curve, saving, approval and default rates, threshold map |
| Thin-file & explainability | Thin-file caveat and counts, AUC by history group, ablation, SHAP importance and directions, reason codes with waterfalls |
| Fairness | Approval, default and wrongful-rejection rates by gender and age, with vs without gender, proxy discrimination, decision |
| Live risk predictor | 13-field form, three real example applicants, risk gauge, decision at the cost-optimal threshold, top 5 SHAP reasons |
| Conclusion & future scope | Answers to the research questions with numbers, limitations, future work, references, about |

Every number on the website is read from `app/assets/` (built from the notebook outputs); no result is typed into the
page code.

## Reproducibility and leakage controls

- `random_state = 42` everywhere; LightGBM runs with `deterministic=True`.
- The 80/20 stratified holdout is drawn **once** by notebook 03 and saved to `data/processed/split_ids.parquet`;
  later runs reload it. Delete that file only if you deliberately want a new split.
- The holdout is never used for fitting, tuning, imputation, scaling, early stopping or threshold selection; the
  decision threshold is chosen on out-of-fold training predictions.
- `application_test.csv` has no labels and is not used for evaluation.
- The website bundle `app/assets/` contains only aggregates and model outputs without identifiers.
