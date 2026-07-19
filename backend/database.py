"""
SQLite + SQLAlchemy setup.

Single responsibility: create the engine, the session factory, and the
declarative Base. Tables are auto-created via Base.metadata.create_all()
(called on app startup) -- no Alembic, no migrations (MVP).
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# DB file lives next to the backend package; overridable via env var.
DB_PATH = os.getenv("PMS_DB_PATH", os.path.join(os.path.dirname(__file__), "pms.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False is required for SQLite under FastAPI's threadpool.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Create all tables if they do not exist."""
    # Import models so they register on Base before create_all.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yield a session and always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
