"""
FastAPI application entry point.

Creates the app, initialises the SQLite database on startup, and registers
routes. Run with:  uvicorn backend.main:app --reload
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routes import router

# Load environment variables (e.g. GEMINI_API_KEY) from .env if present.
load_dotenv()

app = FastAPI(
    title="Predictive Maintenance System API",
    description="MVP backend: upload sensor snapshots, predict failure risk, store results.",
    version="0.1.0",
)

# Allow the frontend (any origin during the hackathon MVP) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


app.include_router(router)
