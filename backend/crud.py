"""
Database read/write helpers.

Separates DB access from route logic so endpoints stay thin.
"""

import json
from typing import List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from . import models
from .ml_service import FEATURES


def store_machines(db: Session, df: pd.DataFrame) -> List[models.Machine]:
    """Persist uploaded snapshots into the machines table."""
    records = []
    for _, row in df.iterrows():
        machine = models.Machine(
            machine_id=str(row["machine_id"]),
            machine_type=str(row["machine_type"]),
            temperature=float(row["temperature"]),
            vibration=float(row["vibration"]),
            current=float(row["current"]),
            load=float(row["load"]),
            operating_hours=float(row["operating_hours"]),
            maintenance_count=int(row["maintenance_count"]),
            ambient_temperature=float(row["ambient_temperature"]),
            humidity=float(row["humidity"]),
        )
        db.add(machine)
        records.append(machine)
    db.commit()
    return records


def get_machines(db: Session, machine_id: Optional[str] = None) -> List[models.Machine]:
    query = db.query(models.Machine)
    if machine_id:
        query = query.filter(models.Machine.machine_id == machine_id)
    return query.all()


def machines_to_dataframe(machines: List[models.Machine]) -> pd.DataFrame:
    """Convert ORM rows to the DataFrame the model expects."""
    rows = [
        {"machine_id": m.machine_id, **{f: getattr(m, f) for f in FEATURES}}
        for m in machines
    ]
    return pd.DataFrame(rows)


def store_prediction(db: Session, result: dict) -> models.Prediction:
    """Persist a single prediction result (top_features as JSON string)."""
    pred = models.Prediction(
        machine_id=result["machine_id"],
        prediction=result["prediction"],
        risk_label=result["risk_label"],
        confidence=result["confidence"],
        top_features=json.dumps(result["top_features"]),
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def get_predictions(db: Session) -> List[models.Prediction]:
    return db.query(models.Prediction).order_by(models.Prediction.created_at.desc()).all()
