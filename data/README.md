# Data

## Source

| | |
|---|---|
| Dataset | Home Credit Default Risk (official Kaggle competition) |
| URL | https://www.kaggle.com/competitions/home-credit-default-risk |
| Data provider | Home Credit Group |
| Download date | 2026-10-27 |
| Kaggle file version | All files dated 2019-12-11 in the Kaggle API listing |
| Downloaded with | Kaggle CLI 2.2.5 (`kaggle competitions download -c home-credit-default-risk -f <file> -p data/raw`) |

The competition rules on Kaggle govern how this data may be used. The raw and processed files are not
committed to this repository (see `.gitignore`), so anyone reproducing the project must accept the
competition rules and download the data themselves.

**Country of origin:** the provider does not disclose which country the data comes from. Currency
amounts (`AMT_*`) are in an unspecified local currency.

## Files downloaded (`data/raw/`)

| File | Rows | Columns | CSV size | Contents |
|---|---:|---:|---:|---|
| `application_train.csv` | 307,511 | 122 | 158.4 MB | One row per loan application, with the `TARGET` label |
| `application_test.csv` | 48,744 | 121 | 25.3 MB | Same as train, without `TARGET` (competition test set) |
| `bureau.csv` | 1,716,428 | 17 | 162.1 MB | The applicant's previous credits at other institutions, as reported to the Credit Bureau |
| `previous_application.csv` | 1,670,214 | 37 | 386.2 MB | The applicant's previous applications to Home Credit |
| `installments_payments.csv` | 13,605,401 | 8 | 689.6 MB | Repayment history for previous Home Credit loans |
| `HomeCredit_columns_description.csv` | 219 | 5 | 37 KB | Column dictionary for all tables. **Not UTF-8:** read it with `encoding="latin-1"` |

Join keys: `SK_ID_CURR` links each table to the application tables. `SK_ID_PREV` links
`installments_payments` to `previous_application`. `SK_ID_BUREAU` identifies rows in `bureau`.

### Target

`TARGET = 1` means the client had payment difficulties (the competition's definition of default).

| TARGET | Count | Share |
|---|---:|---:|
| 0 (repaid) | 282,686 | 91.93% |
| 1 (default) | 24,825 | 8.07% |

The test set has no labels, and no `SK_ID_CURR` appears in both train and test.

## Files intentionally excluded (for now)

These files are part of the competition but were left out of the initial scope. They can be added
later with the same `kaggle competitions download` command.

| File | Size (per Kaggle) | Reason excluded |
|---|---:|---|
| `bureau_balance.csv` | 375.6 MB | Month-by-month status of each Credit Bureau credit. A large, fine-grained history that `bureau.csv` already summarizes at credit level |
| `POS_CASH_balance.csv` | 392.7 MB | Monthly snapshots of previous point-of-sale and cash loans. Large, and not needed for the first modelling pass |
| `credit_card_balance.csv` | 424.6 MB | Monthly snapshots of previous credit cards. Large, and not needed for the first modelling pass |
| `sample_submission.csv` | 0.5 MB | Only shows the Kaggle leaderboard submission format. Not used for analysis |

## Processed files (`data/processed/`)

`data/make_parquet.py` converts each CSV to Parquet with smaller dtypes:

- `float64` → `float32`
- `int64` → the smallest signed integer type that fits (`int8`/`int16`/`int32`)
- text columns with at most 1,000 distinct values (and fewer than half the row count) → `category`

| File | In-memory before | In-memory after | Parquet size |
|---|---:|---:|---:|
| `application_train.parquet` | 325.2 MB | 95.3 MB | 20.6 MB |
| `application_test.parquet` | 51.2 MB | 15.1 MB | 3.7 MB |
| `bureau.parquet` | 271.5 MB | 85.1 MB | 33.1 MB |
| `previous_application.parquet` | 671.8 MB | 146.5 MB | 54.6 MB |
| `installments_payments.parquet` | 830.4 MB | 389.3 MB | 216.6 MB |
| `HomeCredit_columns_description.parquet` | <0.1 MB | <0.1 MB | <0.1 MB |

"Before" is the size of the CSV as loaded by pandas 3.0.5, which stores text as Arrow-backed `str`.
Older pandas versions load text as `object`, so their "before" figures are larger.

Notes:

- `float32` keeps about 7 significant digits. A spot check on `application_train` found the cast
  exact for `AMT_CREDIT`, `AMT_ANNUITY` and `AMT_GOODS_PRICE`, and within 0.0125 (relative error
  under 1e-7) for `AMT_INCOME_TOTAL`. All ID columns stay integers and are exact.
- `HomeCredit_columns_description` has a leftover unnamed index column (`Unnamed: 0`) from the
  original CSV. It was kept as is.

## Derived files (created by the notebooks, also in `data/processed/`)

| File | Created by | Contents |
|---|---|---|
| `model_table.parquet` | `notebooks/02_feature_engineering.ipynb` | One row per labelled applicant (307,507 × 163): cleaned application columns, ratios, `BUR_`/`PREV_`/`INST_` aggregates, history flags, `TARGET` |
| `split_ids.parquet` | `notebooks/03_modeling.ipynb` | `SK_ID_CURR` and `SET` (train / holdout) for the stratified 80/20 split, drawn once |
| `oof_lgbm.parquet` | `notebooks/03_modeling.ipynb` | Out-of-fold LightGBM predictions for the training applicants (used to choose the threshold) |
| `holdout_predictions.parquet` | `notebooks/03_modeling.ipynb` | Holdout predictions of all three models |

## Reproduce

```bash
pip install kaggle pandas pyarrow
# requires ~/.kaggle/kaggle.json and accepted competition rules
for f in application_train.csv application_test.csv bureau.csv previous_application.csv \
         installments_payments.csv HomeCredit_columns_description.csv; do
  kaggle competitions download -c home-credit-default-risk -f "$f" -p data/raw
done
# unzip any *.zip in data/raw, then:
python data/make_parquet.py
```
