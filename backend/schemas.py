"""
Pydantic schemas for request/response validation.

Kept intentionally small (MVP). The single-snapshot schema doubles as our
input validation guardrail before the model is called.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MachineSnapshot(BaseModel):
    """A single machine sensor snapshot for prediction (JSON body)."""

    machine_id: str
    machine_type: str
    temperature: float = Field(ge=0)
    vibration: float = Field(ge=0)
    current: float = Field(ge=0)
    load: float = Field(ge=0, le=100)
    operating_hours: float = Field(ge=0)
    maintenance_count: int = Field(ge=0)
    ambient_temperature: float
    humidity: float = Field(ge=0, le=100)


class PredictionResult(BaseModel):
    """Model output returned to the client."""

    machine_id: str
    prediction: int
    risk_label: str
    confidence: float
    top_features: List[dict]


class UploadResponse(BaseModel):
    rows_stored: int
    machine_ids: List[str]


class PredictionRecord(BaseModel):
    """A stored prediction row (for GET /predictions)."""

    id: int
    machine_id: str
    prediction: int
    risk_label: str
    confidence: float
    top_features: str
    created_at: datetime

    class Config:
        from_attributes = True


class PredictRequest(BaseModel):
    """Optional filter for the stored-machines prediction flow."""

    machine_id: Optional[str] = None


class ExplainRequest(BaseModel):
    """
    Payload for a GenAI explanation. Sensor values are optional and, when
    provided, enable sensor-anomaly detection in the explanation layer. They are
    used only for the explanation -- the ML prediction is unchanged.
    """

    machine_id: str
    risk_label: str
    confidence: float = Field(ge=0, le=1)
    top_features: List[dict] = []
    # Optional sensor snapshot (for anomaly detection in the explanation).
    temperature: Optional[float] = None
    vibration: Optional[float] = None
    current: Optional[float] = None
    load: Optional[float] = None
    operating_hours: Optional[float] = None
    maintenance_count: Optional[float] = None
    ambient_temperature: Optional[float] = None
    humidity: Optional[float] = None


class ExplainResponse(BaseModel):
    machine_id: str
    explanation: str
    source: str
    disclaimer: str
