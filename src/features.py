# src/features.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_FILE = INTERIM_DIR / "sample_accidents.csv"
PROCESSED_FILE = PROCESSED_DIR / "processed_sample.csv"

# --------------- Helpers / Transformers ---------------

class DateTimeFeaturizer(BaseEstimator, TransformerMixin):
    """Creates hour/day/month/duration/is_weekend features from Start_Time and End_Time (if present)."""
    def __init__(self, start_col="Start_Time", end_col="End_Time"):
        self.start_col = start_col
        self.end_col = end_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # parse start time
        if self.start_col in X.columns:
            X[self.start_col] = pd.to_datetime(X[self.start_col], errors="coerce")
            X["start_hour"] = X[self.start_col].dt.hour
            X["start_dayofweek"] = X[self.start_col].dt.dayofweek
            X["start_month"] = X[self.start_col].dt.month
            X["is_weekend"] = X[self.start_col].dt.dayofweek.isin([5, 6]).astype(int)
        else:
            # create empty columns if not present
            X["start_hour"] = np.nan
            X["start_dayofweek"] = np.nan
            X["start_month"] = np.nan
            X["is_weekend"] = np.nan

        # duration (minutes)
        if self.start_col in X.columns and self.end_col in X.columns:
            X[self.end_col] = pd.to_datetime(X[self.end_col], errors="coerce")
            X["duration_min"] = (X[self.end_col] - X[self.start_col]).dt.total_seconds() / 60.0
        else:
            X["duration_min"] = np.nan

        return X

# --------------- Build preprocessor ---------------

def pick_candidate_columns(df: pd.DataFrame):
    """Return sensible numeric and categorical column lists based on available columns."""
    numeric_candidates = [
        "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)", "Start_Lat", "Start_Lng",
        "start_hour", "start_dayofweek", "start_month", "duration_min"
    ]
    categorical_candidates = [
        "Weather_Condition", "Sunrise_Sunset", "Amenity", "Bump",
        "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
        "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal",
        "Wind_Direction", "Side", "City", "County", "State"
    ]

    numeric = [c for c in numeric_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns]
    return numeric, categorical

def build_preprocessor(df_sample: pd.DataFrame):
    numeric_cols, categorical_cols = pick_candidate_columns(df_sample)

    # numeric pipeline: impute (median) then scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # categorical pipeline: impute (constant) then one-hot encode
    cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop"  # drop other columns
    )

    return preprocessor, numeric_cols + categorical_cols

# --------------- Runner to create processed CSV ---------------

def create_processed_sample(sample_path: Path = SAMPLE_FILE, out_path: Path = PROCESSED_FILE, n_rows: int | None = None):
    print(f"Reading sample: {sample_path}")
    df = pd.read_csv(sample_path)
    print("Initial shape:", df.shape)

    fe = DateTimeFeaturizer(start_col="Start_Time", end_col="End_Time")
    df = fe.transform(df)

    # select a target column name if present
    target_col = None
    for candidate in ["Severity", "severity", "Accident_Severity"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        print("⚠️ No target column found (Severity). You may need to adjust the target name manually.")
    else:
        print("Target column detected:", target_col)

    # optionally reduce rows for quick experiments
    if n_rows is not None and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=42).reset_index(drop=True)
        print("Sampled down to:", df.shape)

    numeric, categorical = pick_candidate_columns(df)
    print("Numeric features used:", numeric)
    print("Categorical features used:", categorical)

    # Save processed csv with the original columns plus created features
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("✅ Processed file saved to:", out_path)
    return df

# --------------- If executed directly ---------------

if __name__ == "__main__":
    # create smaller processed file (keep default full sample or pass n_rows)
    create_processed_sample(n_rows=50000)  # keep at 50k for fast experiments; change or set None
