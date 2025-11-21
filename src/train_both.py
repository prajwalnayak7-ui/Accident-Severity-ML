# src/train_both.py
import json
import time
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

# XGBoost import
from xgboost import XGBClassifier

# local preprocessor builder
from .features import build_preprocessor

# ----------------------------
# Config
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "processed_sample.csv"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
RANDOM_STATE = 42

# Ensure models dir exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed(path: Path = PROCESSED_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed file not found at: {path}")
    return pd.read_csv(path)


def _evaluate_and_save(pipe, X_test, y_test, model_name: str):
    """Evaluate pipeline on test and return metrics dict."""
    print(f"\nEvaluating {model_name} on test set...")
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    clf_report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()

    print(f"{model_name} Accuracy: {acc:.4f}")
    print(f"{model_name} F1 (macro): {f1_macro:.4f}")
    print(f"{model_name} F1 (weighted): {f1_weighted:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:")
    print(conf_mat)

    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": conf_mat,
        "classification_report": clf_report,
    }
    return metrics


def train_single_model(pipe_template, param_dist, X_train, y_train, X_test, y_test, model_name, n_iter=8):
    """Run RandomizedSearchCV on a template pipeline and return fitted best_estimator_ and metrics."""
    print(f"\n--- Training {model_name} ---")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        pipe_template,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"-> {model_name} search done in {elapsed/60:.2f} minutes. Best params:")
    print(search.best_params_)

    best_pipe = search.best_estimator_

    metrics = _evaluate_and_save(best_pipe, X_test, y_test, model_name)
    return best_pipe, search.best_params_, metrics


def main():
    print("Loading processed data...")
    df = load_processed()
    print("Data shape:", df.shape)

    target = "Severity"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in processed data.")

    y = df[target]

# XGBoost requires classes starting at 0
    label_map = {1: 0, 2: 1, 3: 2, 4: 3}
    y = y.map(label_map)

    X = df.drop(columns=[target])


    print("Building preprocessing pipeline...")
    preprocessor, used_cols = build_preprocessor(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # --------------------------
    # Random Forest pipeline
    # --------------------------
    rf_clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    rf_pipe = ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", rf_clf),
        ]
    )

    rf_param_dist = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
    }

    rf_best_pipe, rf_best_params, rf_metrics = train_single_model(
        rf_pipe, rf_param_dist, X_train, y_train, X_test, y_test, model_name="RandomForest", n_iter=6
    )

    # Save RF
    rf_path = MODELS_DIR / "rf_pipeline.joblib"
    joblib.dump(rf_best_pipe, rf_path)
    print(f"Saved RandomForest pipeline to {rf_path}")

    # --------------------------
    # XGBoost pipeline
    # --------------------------
    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgb_pipe = ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", xgb_clf),
        ]
    )

    xgb_param_dist = {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [3, 6, 10],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.6, 0.8, 1.0],
    }

    xgb_best_pipe, xgb_best_params, xgb_metrics = train_single_model(
        xgb_pipe, xgb_param_dist, X_train, y_train, X_test, y_test, model_name="XGBoost", n_iter=8
    )

    # Save XGB
    xgb_path = MODELS_DIR / "xgb_pipeline.joblib"
    joblib.dump(xgb_best_pipe, xgb_path)
    print(f"Saved XGBoost pipeline to {xgb_path}")

    # --------------------------
    # Save combined metrics
    # --------------------------
    combined = {
        "random_forest": {
            "best_params": rf_best_params,
            "metrics": rf_metrics,
            "model_path": str(rf_path),
        },
        "xgboost": {
            "best_params": xgb_best_params,
            "metrics": xgb_metrics,
            "model_path": str(xgb_path),
        },
        "test_set_size": int(len(y_test)),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined metrics to {METRICS_PATH}")
    print("Training finished. You can now compare rf_pipeline.joblib and xgb_pipeline.joblib.")


if __name__ == "__main__":
    main()
