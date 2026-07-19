"""
SQLAlchemy ORM models.

Only the two tables we actually use (MVP): machines (uploaded snapshots) and
predictions (one row per prediction, with confidence + top features).
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from .database import Base


class Machine(Base):
    """A single uploaded sensor snapshot for a machine."""

    __tablename__ = "machines"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    machine_type = Column(String)
    temperature = Column(Float)
    vibration = Column(Float)
    current = Column(Float)
    load = Column(Float)
    operating_hours = Column(Float)
    maintenance_count = Column(Integer)
    ambient_temperature = Column(Float)
    humidity = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """One failure-risk prediction for a machine snapshot."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    prediction = Column(Integer)          # 0 = healthy, 1 = likely failure
    risk_label = Column(String)           # "High" / "Low"
    confidence = Column(Float)            # P(failure), 0..1
    top_features = Column(String)         # JSON string of top-3 drivers
    created_at = Column(DateTime, default=datetime.utcnow)
