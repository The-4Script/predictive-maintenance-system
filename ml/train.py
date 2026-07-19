"""
Train the predictive-maintenance failure classifier.

Primary model: RandomForestClassifier (intentional project decision -- strong
on tabular sensor data, fast on CPU, exposes feature importance for
explainability, trivial to deploy via joblib).

The whole preprocessing + model is wrapped in a single sklearn Pipeline so
inference in the backend is one call: pipeline.predict_proba(new_snapshot).

Usage (from project root):
    python -m ml.train
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as C

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# Columns intentionally EXCLUDED from training:
#   machine_id  -> identifier, not predictive
#   risk_score  -> derived from the target during generation (would leak)
#   failure     -> the target itself
TARGET = "failure"
CATEGORICAL_FEATURES = ["machine_type"]
NUMERIC_FEATURES = [
    "temperature",
    "vibration",
    "current",
    "load",
    "operating_hours",
    "maintenance_count",
    "ambient_temperature",
    "humidity",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MODEL_PATH = "ml/artifacts/model.joblib"
METRICS_PATH = "ml/artifacts/metrics.json"
IMPORTANCE_PATH = "ml/artifacts/feature_importance.json"

# Fixed seed for reproducible splits and forest.
RANDOM_STATE = C.RANDOM_SEED
TEST_SIZE = 0.2


def build_pipeline() -> Pipeline:
    """Preprocessing + Random Forest in a single reproducible Pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",  # respect the ~20/80 imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _expanded_feature_names(pipeline: Pipeline) -> list[str]:
    """Recover feature names after one-hot expansion for importance mapping."""
    ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return cat_names + NUMERIC_FEATURES


def main() -> None:
    df = pd.read_csv(C.OUTPUT_CSV)

    X = df[FEATURES]
    y = df[TARGET]

    # Held-out test set; stratified to preserve the failure ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # ---- Evaluation on the held-out test set ----
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "RandomForestClassifier",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    # ---- Feature importance (explainability) ----
    importances = pipeline.named_steps["model"].feature_importances_
    names = _expanded_feature_names(pipeline)
    importance = sorted(
        ({"feature": n, "importance": round(float(v), 4)} for n, v in zip(names, importances)),
        key=lambda d: d["importance"],
        reverse=True,
    )

    # ---- Persist artifacts ----
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(IMPORTANCE_PATH, "w") as f:
        json.dump(importance, f, indent=2)

    # ---- Honest summary ----
    print(f"Model: RandomForestClassifier  (seed={RANDOM_STATE})")
    print(f"Train/Test: {metrics['n_train']}/{metrics['n_test']}")
    print(f"Accuracy: {metrics['accuracy']}   ROC-AUC: {metrics['roc_auc']}")
    print("Top 5 features:")
    for row in importance[:5]:
        print(f"  {row['feature']:<22} {row['importance']}")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved feature importance -> {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
