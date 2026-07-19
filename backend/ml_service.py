"""
ML inference service.

Loads the trained Random Forest pipeline ONCE and runs predictions. Keeps the
model concern isolated from the web/DB layers. Never sees risk_score/failure
(leakage-safe): it only uses the agreed feature columns.
"""

import json
import os
from functools import lru_cache
from typing import List

import joblib
import pandas as pd

# Feature contract -- must match ml/train.py.
CATEGORICAL_FEATURES = ["machine_type"]
NUMERIC_FEATURES = [
    "temperature", "vibration", "current", "load",
    "operating_hours", "maintenance_count", "ambient_temperature", "humidity",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Model + importance artifacts (overridable via env for deployment).
_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "ml", "artifacts", "model.joblib")
_DEFAULT_IMPORTANCE = os.path.join(
    os.path.dirname(__file__), "..", "ml", "artifacts", "feature_importance.json"
)
MODEL_PATH = os.getenv("PMS_MODEL_PATH", _DEFAULT_MODEL)
IMPORTANCE_PATH = os.getenv("PMS_IMPORTANCE_PATH", _DEFAULT_IMPORTANCE)

RISK_THRESHOLD = 0.5  # P(failure) at/above this => High risk (predicted failure)


def risk_label_from_proba(p: float) -> str:
    """
    Map a failure probability to a two-tier risk label.

    We use only High / Low because the ML target is binary (fail / not-fail):
    the same 0.5 threshold that decides the prediction also decides the label,
    so 'risk = the model's decision' and 'confidence = how sure it is'. A Medium
    band would be an arbitrary slice of probability with no clear operational
    meaning (inspect or don't?), so we removed it.
    """
    return "High" if p >= RISK_THRESHOLD else "Low"


@lru_cache(maxsize=1)
def _load_model():
    """Load and cache the pipeline once per process."""
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _global_top_features() -> tuple:
    """Model-level top-3 feature importances (fallback explanation)."""
    try:
        with open(IMPORTANCE_PATH) as f:
            imp = json.load(f)
        return tuple(imp[:3])
    except FileNotFoundError:
        return tuple()


def is_model_ready() -> bool:
    return os.path.exists(MODEL_PATH)


def predict_dataframe(df: pd.DataFrame) -> List[dict]:
    """
    Run predictions for a DataFrame of snapshots. Returns one result dict per
    row with prediction, risk_label, confidence, and top_features.
    """
    model = _load_model()
    X = df[FEATURES]
    proba = model.predict_proba(X)[:, 1]

    top = [{"feature": d["feature"], "importance": d["importance"]} for d in _global_top_features()]

    results = []
    for i, p in enumerate(proba):
        pred = int(p >= RISK_THRESHOLD)
        results.append(
            {
                "machine_id": str(df.iloc[i].get("machine_id", f"row-{i}")),
                "prediction": pred,
                "risk_label": risk_label_from_proba(float(p)),
                "confidence": round(float(p), 4),
                "top_features": top,
            }
        )
    return results
