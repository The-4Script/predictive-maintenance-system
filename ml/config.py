"""
Configuration for synthetic dataset generation.

All tunable parameters live here so the generation logic stays free of
hardcoded values. Teammates can adjust machine behaviour, degradation speed,
and failure thresholds without touching the generator code.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset size
# ---------------------------------------------------------------------------
# Target ~3000 snapshots spread across a small fleet of machines (time-series).
NUM_MACHINES = 15
SNAPSHOTS_PER_MACHINE = 200  # 15 * 200 = 3000 records
TARGET_FAILURE_RATIO = 0.20  # ~20% of snapshots should be failures

# ---------------------------------------------------------------------------
# Machine types and their HEALTHY baseline sensor profiles.
# Each entry: (mean, std) describing normal operating values for a new/healthy
# machine of that type. Degradation is added on top of these baselines.
# ---------------------------------------------------------------------------
MACHINE_TYPES = {
    "CNC": {
        "temperature": (55.0, 3.0),   # deg C
        "vibration": (2.0, 0.3),      # mm/s (RMS)
        "current": (18.0, 1.5),       # Amps
        "load": (60.0, 8.0),          # % of rated load
    },
    "Pump": {
        "temperature": (50.0, 3.0),
        "vibration": (2.5, 0.4),
        "current": (14.0, 1.2),
        "load": (65.0, 7.0),
    },
    "Motor": {
        "temperature": (60.0, 4.0),
        "vibration": (1.8, 0.3),
        "current": (22.0, 2.0),
        "load": (70.0, 6.0),
    },
    "Compressor": {
        "temperature": (65.0, 4.0),
        "vibration": (3.0, 0.5),
        "current": (25.0, 2.0),
        "load": (75.0, 6.0),
    },
}

# ---------------------------------------------------------------------------
# Environmental signals (shared across machine types, mild influence).
# ---------------------------------------------------------------------------
AMBIENT_TEMPERATURE = (28.0, 4.0)  # deg C (mean, std)
HUMIDITY = (55.0, 12.0)            # % relative humidity (mean, std)

# ---------------------------------------------------------------------------
# Operating hours: each machine ages over its snapshot series.
# ---------------------------------------------------------------------------
HOURS_START_RANGE = (500, 5000)   # random starting age per machine
HOURS_STEP_RANGE = (40, 80)       # hours added per snapshot (with jitter)

# ---------------------------------------------------------------------------
# Degradation model.
# A fraction of machines are "failing" units: late in their life their
# temperature/vibration/current drift upward, eventually crossing into failure.
# Degradation is expressed as a fraction added to the healthy baseline value.
# ---------------------------------------------------------------------------
DEGRADATION = {
    # portion of a failing machine's series (from the end) that shows drift
    "onset_fraction_range": (0.30, 0.55),
    # peak multiplicative drift applied at end of life for each signal
    "temperature_peak": 0.45,   # up to +45% over baseline
    "vibration_peak": 1.20,     # up to +120% over baseline (bearings dominate)
    "current_peak": 0.30,       # up to +30% over baseline
    "load_peak": 0.15,          # up to +15% over baseline
}

# maintenance_count grows with age; a reset can occur after servicing.
MAINTENANCE_PER_1000H = 1.2  # avg servicing events per 1000 operating hours

# ---------------------------------------------------------------------------
# Failure labelling (rule-based risk score, NOT random).
# Each risky signal contributes a normalised, weighted amount to a risk score.
# Weights reflect engineering intuition: vibration & temperature dominate.
# ---------------------------------------------------------------------------
RISK_WEIGHTS = {
    "temperature": 0.28,
    "vibration": 0.34,
    "current": 0.16,
    "load": 0.10,
    "operating_hours": 0.07,
    "ambient_temperature": 0.03,
    "humidity": 0.02,
}

# Small Gaussian noise on the risk score so the decision boundary is learnable
# but not perfectly separable.
RISK_NOISE_STD = 0.04

# Output location
OUTPUT_CSV = "data/synthetic_sensor_data.csv"
