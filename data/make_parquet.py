"""Convert the raw Home Credit CSVs in data/raw/ to compact Parquet files in data/processed/.

Dtype reductions applied to every file:
  - float64 -> float32
  - int64   -> smallest signed integer type that holds the column's range
  - low-cardinality text columns -> category

Usage (from the project root):
    python data/make_parquet.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "HomeCredit_columns_description.csv",
]

# The column description file is not valid UTF-8.
ENCODINGS = {"HomeCredit_columns_description.csv": "latin-1"}

# A text column becomes a category if it has at most MAX_CATEGORIES distinct
# values and those make up less than MAX_UNIQUE_RATIO of the rows.
MAX_CATEGORIES = 1000
MAX_UNIQUE_RATIO = 0.5


def to_mb(n_bytes):
    return n_bytes / 1024**2


def is_text(s):
    # pandas >= 3 reads text as StringDtype ("str"); older versions use object.
    return s.dtype == object or isinstance(s.dtype, pd.StringDtype)


def shrink(df):
    for col in df.columns:
        s = df[col]
        if s.dtype == "float64":
            df[col] = s.astype("float32")
        elif s.dtype == "int64":
            df[col] = pd.to_numeric(s, downcast="integer")
        elif is_text(s):
            n_unique = s.nunique(dropna=True)
            if n_unique <= MAX_CATEGORIES and n_unique < MAX_UNIQUE_RATIO * len(s):
                df[col] = s.astype("category")
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    header = f"{'file':<36}{'rows':>12}{'cols':>6}{'before MB':>11}{'after MB':>10}{'saved':>7}{'parquet MB':>12}"
    print(header)
    print("-" * len(header))
    for name in FILES:
        df = pd.read_csv(RAW_DIR / name, encoding=ENCODINGS.get(name, "utf-8"))
        before = df.memory_usage(deep=True).sum()
        df = shrink(df)
        after = df.memory_usage(deep=True).sum()

        out_path = PROCESSED_DIR / name.replace(".csv", ".parquet")
        df.to_parquet(out_path, index=False)

        print(
            f"{name:<36}{len(df):>12,}{df.shape[1]:>6}"
            f"{to_mb(before):>11.1f}{to_mb(after):>10.1f}{1 - after / before:>7.0%}"
            f"{to_mb(out_path.stat().st_size):>12.1f}"
        )


if __name__ == "__main__":
    main()
