# ML Pipeline — Predictive Maintenance

CPU-friendly ML pipeline: synthetic data generation → Random Forest training →
saved inference model + explainability artifacts.

## Layout

| File | Purpose |
|------|---------|
| `config.py` | All tunable parameters (ranges, weights, degradation, seed). |
| `data_generator.py` | Core synthetic-data logic (importable by the backend). |
| `generate_dataset.py` | Runnable: writes the seed CSV. |
| `train.py` | Trains the Random Forest, evaluates, saves artifacts. |
| `artifacts/` | `model.joblib`, `metrics.json`, `feature_importance.json`. |

## Commands (run from project root)

```bash
pip install -r requirements.txt

# 1. Regenerate the synthetic dataset (reproducible, seed in config.py)
python -m ml.generate_dataset      # -> data/synthetic_sensor_data.csv

# 2. Train the model (reproducible, held-out stratified test set)
python -m ml.train                 # -> ml/artifacts/*
```

A ready-to-use seed dataset is committed at `data/synthetic_sensor_data.csv`,
so the project can be trained immediately without regenerating.

## Model

- **RandomForestClassifier** (intentional choice: strong on tabular sensor
  data, fast on CPU, exposes feature importance, simple joblib deployment).
- Wrapped in an sklearn **Pipeline** (`OneHotEncoder(machine_type)` +
  passthrough numeric sensors → forest), so inference is a single call.
- `class_weight="balanced"` handles the ~20/80 failure imbalance.
- Fixed `random_state` (from `config.RANDOM_SEED`) for reproducibility.

### Leakage safety
Training features are the **real sensor + machine attributes only**.
`machine_id` (identifier), `risk_score` (derived from the target during
generation), and `failure` (the target) are **excluded** from training.
`risk_score` is retained in the CSV for analysis/explainability only.

### Metrics — honesty note
On synthetic data the model scores very high (accuracy ~0.98, ROC-AUC ~0.997).
This is expected: the data is generated from a known degradation process, so it
is more separable than real-world sensor data. We report these as
**sanity-check metrics on synthetic data**, not a claim of real-world accuracy.
Live metrics are read from `ml/artifacts/metrics.json`.

### Explainability
`ml/artifacts/feature_importance.json` ranks features by Random Forest
importance. Top drivers are vibration, operating_hours, and temperature —
consistent with the engineering-inspired degradation model.
