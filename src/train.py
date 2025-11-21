# src/train.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

# local feature utilities (expects src.features to be importable)
from .features import build_preprocessor

# ----------------------------
# Configuration / constants
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "processed_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "pipeline.joblib"
RANDOM_STATE = 42


def load_processed(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at: {path}")
    return pd.read_csv(path)


def train_model():
    print("📂 Loading processed data...")
    df = load_processed()
    print("Shape:", df.shape)

    target = "Severity"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in processed data.")

    y = df[target]
    X = df.drop(columns=[target])

    print("🧭 Building preprocessing pipeline...")
    preprocessor, used_cols = build_preprocessor(df)  # expects (transformer, list_of_cols)

    print("🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # base classifier
    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    # end-to-end pipeline: preprocessing -> SMOTE -> classifier
    pipeline = ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ]
    )

    # hyperparameter grid for RandomizedSearch
    param_distributions = {
        "clf__n_estimators": [100, 200, 400, 600],
        "clf__max_depth": [None, 10, 20, 30],
        "clf__min_samples_split": [2, 5, 10],
    }

    print("🔍 Running RandomizedSearchCV (this may take a while)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    search.fit(X_train, y_train)

    print("✅ Training complete!")
    print("Best parameters:", search.best_params_)

    best_pipe = search.best_estimator_

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    y_pred = best_pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    clf_report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(conf_mat)

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save best pipeline
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

    # Save metrics to JSON so UI / API can read them
    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": conf_mat,
        "classification_report": clf_report,
        "best_params": search.best_params_,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train_model()
