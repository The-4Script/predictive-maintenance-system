"""
API endpoints (thin controllers).

  GET  /health           liveness + model-loaded check
  POST /upload           CSV -> validate -> store in machines
  POST /predict          predict on stored machines -> store predictions
  POST /predict/single   predict one snapshot from JSON body
  GET  /predictions      prediction history
"""

import io
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from . import crud, schemas
from .cost_service import fleet_impact, per_machine_impact
from .database import get_db
from .genai_service import generate_explanation, is_genai_enabled
from .ml_service import FEATURES, is_model_ready, predict_dataframe

router = APIRouter()

REQUIRED_COLUMNS = ["machine_id"] + FEATURES


@router.get("/health")
def health():
    return {"status": "ok", "model_ready": is_model_ready(), "genai_enabled": is_genai_enabled()}


@router.post("/upload", response_model=schemas.UploadResponse)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Accept a CSV, validate required columns, store snapshots."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # Basic guardrail: reject non-numeric / obviously invalid sensor values.
    try:
        df[FEATURES[1:]] = df[FEATURES[1:]].apply(pd.to_numeric)  # numeric features
    except Exception:
        raise HTTPException(status_code=400, detail="Sensor columns must be numeric")

    stored = crud.store_machines(db, df)
    return schemas.UploadResponse(
        rows_stored=len(stored),
        machine_ids=[m.machine_id for m in stored],
    )


@router.post("/predict", response_model=List[schemas.PredictionResult])
def predict(req: schemas.PredictRequest, db: Session = Depends(get_db)):
    """Predict on machines already stored via /upload."""
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not available")

    machines = crud.get_machines(db, req.machine_id)
    if not machines:
        raise HTTPException(status_code=404, detail="No stored machines to predict")

    df = crud.machines_to_dataframe(machines)
    results = predict_dataframe(df)

    for r in results:
        crud.store_prediction(db, r)
    return results


@router.post("/predict/single", response_model=schemas.PredictionResult)
def predict_single(snapshot: schemas.MachineSnapshot):
    """
    Predict a single snapshot from a JSON body (demo/testing friendly).

    Not persisted by default -- keeps prediction history clean during testing.
    The CSV workflow (/predict) is the flow that logs to SQLite.
    """
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not available")

    df = pd.DataFrame([snapshot.model_dump()])
    return predict_dataframe(df)[0]


@router.get("/api/machines")
def api_machines(db: Session = Depends(get_db)):
    """
    Return stored machines with their predicted risk (for the dashboard table).
    Runs the model on-the-fly so every stored machine has a risk label + the
    sensor values needed for the details view.
    """
    machines = crud.get_machines(db)
    if not machines:
        return []

    df = crud.machines_to_dataframe(machines)
    results = predict_dataframe(df)

    # Merge sensor values with prediction results, keyed by row order.
    merged = []
    for m, r in zip(machines, results):
        merged.append(
            {
                "machine_id": m.machine_id,
                "machine_type": m.machine_type,
                "temperature": m.temperature,
                "vibration": m.vibration,
                "current": m.current,
                "load": m.load,
                "operating_hours": m.operating_hours,
                "maintenance_count": m.maintenance_count,
                "ambient_temperature": m.ambient_temperature,
                "humidity": m.humidity,
                "risk_label": r["risk_label"],
                "confidence": r["confidence"],
                "top_features": r["top_features"],
                "impact": per_machine_impact(r["risk_label"], r["confidence"]),
            }
        )
    return merged


@router.get("/api/impact")
def api_impact(db: Session = Depends(get_db)):
    """
    Fleet-level business impact: unplanned downtime avoided + cost saved.
    Computed from current predictions on stored machines.
    """
    machines = crud.get_machines(db)
    if not machines:
        return fleet_impact([])
    df = crud.machines_to_dataframe(machines)
    results = predict_dataframe(df)
    return fleet_impact(results)


@router.get("/predictions", response_model=List[schemas.PredictionRecord])
def list_predictions(db: Session = Depends(get_db)):
    return crud.get_predictions(db)


@router.post("/explain", response_model=schemas.ExplainResponse)
def explain(req: schemas.ExplainRequest):
    """
    GenAI engineering-grade explanation + prioritized recommendation.

    Guardrails: only this machine's summary + sensor readings are sent to the
    LLM (never the database). Detects impossible sensor values (sensor faults)
    vs genuine machine degradation. Always returns a disclaimer; falls back to a
    deterministic report if no API key is configured.
    """
    # Collect the optional sensor snapshot (used only for the explanation).
    sensors = {
        "temperature": req.temperature,
        "vibration": req.vibration,
        "current": req.current,
        "load": req.load,
        "operating_hours": req.operating_hours,
        "maintenance_count": req.maintenance_count,
        "ambient_temperature": req.ambient_temperature,
        "humidity": req.humidity,
    }
    sensors = {k: v for k, v in sensors.items() if v is not None}

    result = generate_explanation(
        machine_id=req.machine_id,
        risk_label=req.risk_label,
        confidence=req.confidence,
        top_features=req.top_features,
        sensors=sensors or None,
    )
    return schemas.ExplainResponse(**result)
