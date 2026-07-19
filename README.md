# Predictive Maintenance System (PEMS)

AI-powered predictive maintenance MVP — synthetic sensor data → Random Forest →
FastAPI + SQLite → GenAI explanation → dashboard.

🏆 **4th Prize — Cognizant Hackathon 2026** · Team 4Script

For maintenance engineers: predicts which industrial machines are likely to
fail soon, with a confidence score, a plain-language reason, and the rupee
value of acting early — so breakdowns are prevented before they happen. The
AI only recommends; the engineer always makes the final decision.

## Architecture

```
Synthetic Data Generator → CSV Dataset (3000 rows) → Random Forest → model.joblib
Engineer uploads CSV → FastAPI Backend → SQLite (machines + predictions)
                     → Prediction + Confidence + Top Factors
                     → GenAI Explanation (Gemini, guardrailed)
                     → Business Impact (downtime + cost saved)
                     → Dashboard UI → Engineer reviews & Approves Work Order
```

- `ml/` — synthetic data generator + Random Forest training (see [ml/README.md](ml/README.md))
- `backend/` — FastAPI app, SQLite (machines + predictions), ML + GenAI services
- `frontend/` — single-file dashboard (upload, predict, explanation, history), no build step

Three golden rules: the frontend never touches the database directly (only
through the API), the LLM is never the source of truth (ML predicts, GenAI
only explains), and the model loads once at startup.

See [explain.md](explain.md) for the full design writeup — ML metrics, risk
classification logic, cost formula, guardrails, and demo Q&A.

## Quick start

```bash
pip install -r requirements.txt

# 1. (optional) regenerate the full dataset + retrain the model
python -m ml.generate_dataset
python -m ml.train

# 2. start the backend API
uvicorn backend.main:app --reload   # http://127.0.0.1:8000/docs

# 3. open the frontend
# just open frontend/index.html in your browser
```

## Datasets

- `data/synthetic_sensor_data.csv` — the full generated dataset (3000 rows,
  15 machines) used to train the model. Regenerate any time with
  `python -m ml.generate_dataset`.
- `TESTING_DATA/synthetic_sensor_data.csv` — a small ~20-25 row sample
  covering the interesting edge cases (normal, degrading, and anomalous
  readings). Upload this file in the dashboard's "Upload machine snapshots"
  step for a clean demo run.

## GenAI (optional)

Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`. Without a key,
the app uses a safe fallback explanation template (demo still works).

## Guardrails

Token-optimized LLM calls (only prediction + top-3 factors sent), scope-locked
prompt, recommend-only (no final decisions), disclaimer shown on all AI output,
input validation, audit log via prediction history.

## Team 4Script

- [Kaustubh Bhoir](https://www.linkedin.com/in/kaustubh-bhoir-ce/)
- [Durvesh Thorat](https://www.linkedin.com/in/durvesh-thorat/)
- [Nipun Tamore](https://www.linkedin.com/in/nipun-tamore-21ba5b308/)
- [Arnav Patil](https://www.linkedin.com/in/arnav-pradip-patil-3b872b358/)
