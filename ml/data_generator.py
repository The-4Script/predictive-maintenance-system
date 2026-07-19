"""
Synthetic predictive-maintenance dataset generator.

Design goals:
  * Time-series per machine: each machine ages over increasing operating_hours.
  * Correlated degradation: for failing machines, temperature, vibration,
    current and load rise TOGETHER as the machine deteriorates late in life.
  * Engineering-rule failure labels (NOT random): a snapshot is a failure when
    sensor values cross realistic thresholds, when a machine is old and
    neglected, when several signals are jointly abnormal, or when the continuous
    risk_score is high.
  * Edge cases: ~4% of rows carry extreme/impossible sensor values (e.g.
    temperature = 6000). These are always labelled failure so the trained model
    LEARNS that extreme regions mean failure -- a Random Forest cannot
    extrapolate beyond the values it was trained on, so those examples must
    exist in the data.

Public interface is unchanged: `generate_dataset(seed)` returns a DataFrame with
exactly the agreed schema. config.py, train.py and generate_dataset.py are
untouched and remain compatible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# ---------------------------------------------------------------------------
# Generator-local tuning constants (kept here, not in config.py, so config
# stays untouched). These control how strongly a failing machine degrades and
# how often anomaly rows are injected.
# ---------------------------------------------------------------------------
# Additive end-of-life drift applied on top of the healthy baseline as a
# failing machine reaches the end of its series (progress = 1.0).
DRIFT_TEMPERATURE = 32.0   # deg C added at full degradation
DRIFT_VIBRATION = 3.5      # mm/s added
DRIFT_CURRENT = 12.0       # Amps added
DRIFT_LOAD = 22.0          # % load added

ANOMALY_RATE = 0.04        # ~4% of snapshots are extreme-value anomalies

# Extreme values used for injected anomalies (impossible / sensor-fault region).
ANOMALY_PATTERNS = [
    {"temperature": 6000.0},   # totally broken sensor / catastrophic reading
    {"temperature": 180.0},    # severe overheating
    {"vibration": 18.0},       # violent mechanical vibration
    {"current": 120.0},        # electrical overload
    {"load": 150.0},           # impossible overload
]


def _clip_positive(value: float) -> float:
    """Sensor readings cannot be negative (but extreme highs are kept)."""
    return max(value, 0.0)


def _normalise_signal(value: float, mean: float, std: float) -> float:
    """
    Convert a raw reading into a 0..1 'how abnormal' score relative to its
    healthy baseline, using a saturating function of standard deviations above
    the mean. Values at or below the mean contribute ~0 risk.
    """
    z = (value - mean) / std if std > 0 else 0.0
    if z <= 0:
        return 0.0
    return float(min(z / 3.0, 1.0))  # 3+ std above mean approaches full risk


def _generate_machine_series(
    rng: np.random.Generator,
    machine_index: int,
    machine_type: str,
    is_failing: bool,
) -> list[dict]:
    """Generate the ordered snapshot series for a single machine."""
    profile = C.MACHINE_TYPES[machine_type]
    n = C.SNAPSHOTS_PER_MACHINE
    machine_id = f"MCH-{machine_type[:3].upper()}-{1000 + machine_index}"

    # Aging: operating hours increase across the series.
    hours = rng.uniform(*C.HOURS_START_RANGE)
    hours_steps = rng.uniform(*C.HOURS_STEP_RANGE, size=n)

    # Degradation onset: for failing machines, drift begins partway through life.
    onset_frac = rng.uniform(*C.DEGRADATION["onset_fraction_range"])
    onset_index = int((1.0 - onset_frac) * n)

    records: list[dict] = []

    for i in range(n):
        hours += hours_steps[i]

        # Degradation progress in [0,1]: 0 before onset, ramping to 1 at end.
        if is_failing and i >= onset_index:
            progress = (i - onset_index) / max(n - onset_index, 1)
        else:
            progress = 0.0

        # Healthy baseline readings (small natural fluctuations).
        temp = rng.normal(*profile["temperature"])
        vib = rng.normal(*profile["vibration"])
        curr = rng.normal(*profile["current"])
        load = rng.normal(*profile["load"])

        # Correlated degradation: all mechanical signals rise together with
        # progress (a degrading machine runs hotter, shakes more, draws more
        # current, and is pushed harder).
        temp += progress * DRIFT_TEMPERATURE
        vib += progress * DRIFT_VIBRATION
        curr += progress * DRIFT_CURRENT
        load += progress * DRIFT_LOAD
        load = min(max(load, 0.0), 100.0)  # load is a percentage

        # Environment fluctuates naturally, independent of machine health.
        ambient = rng.normal(*C.AMBIENT_TEMPERATURE)
        humidity = float(np.clip(rng.normal(*C.HUMIDITY), 0.0, 100.0))

        # Maintenance count grows with accumulated operating hours.
        maintenance_count = int((hours / 1000.0) * C.MAINTENANCE_PER_1000H)

        # Inject rare extreme anomalies (~4%). One sensor jumps to an
        # impossible/fault value; these rows are always labelled failure later.
        if rng.random() < ANOMALY_RATE:
            pattern = ANOMALY_PATTERNS[rng.integers(len(ANOMALY_PATTERNS))]
            for key, val in pattern.items():
                if key == "temperature":
                    temp = val
                elif key == "vibration":
                    vib = val
                elif key == "current":
                    curr = val
                elif key == "load":
                    load = val

        records.append(
            {
                "machine_id": machine_id,
                "machine_type": machine_type,
                "temperature": round(_clip_positive(temp), 2),
                "vibration": round(_clip_positive(vib), 3),
                "current": round(_clip_positive(curr), 2),
                "load": round(_clip_positive(load), 2),
                "operating_hours": round(hours, 1),
                "maintenance_count": maintenance_count,
                "ambient_temperature": round(ambient, 2),
                "humidity": round(humidity, 2),
                # risk_score / failure are filled in after the fleet is built.
            }
        )

    return records


def _compute_risk_score(row: dict, rng: np.random.Generator) -> float:
    """
    Continuous risk score (0..~1) from a weighted blend of how abnormal each
    signal is relative to its healthy baseline, plus age/environment and a
    little noise so the boundary is learnable but not perfectly separable.
    """
    profile = C.MACHINE_TYPES[row["machine_type"]]
    w = C.RISK_WEIGHTS

    score = 0.0
    score += w["temperature"] * _normalise_signal(row["temperature"], *profile["temperature"])
    score += w["vibration"] * _normalise_signal(row["vibration"], *profile["vibration"])
    score += w["current"] * _normalise_signal(row["current"], *profile["current"])
    score += w["load"] * _normalise_signal(row["load"], *profile["load"])

    score += w["operating_hours"] * float(min(row["operating_hours"] / 20000.0, 1.0))
    score += w["ambient_temperature"] * _normalise_signal(
        row["ambient_temperature"], *C.AMBIENT_TEMPERATURE
    )
    score += w["humidity"] * _normalise_signal(row["humidity"], *C.HUMIDITY)

    score += rng.normal(0.0, C.RISK_NOISE_STD)
    return float(max(score, 0.0))


def _engineering_failure_rules(df: pd.DataFrame) -> pd.Series:
    """
    Boolean mask of snapshots that are failures by explicit engineering rules.

    These are the DEFINITE failures: extreme sensor values, impossible readings
    (injected anomalies), old-and-neglected machines, and jointly-abnormal
    signals. They guarantee the model sees "extreme region => failure".
    """
    return (
        # Critical single-sensor thresholds (clearly unsafe operation).
        (df["temperature"] > 100)
        | (df["vibration"] > 6)
        | (df["current"] > 32)
        | (df["load"] >= 98)
        # Impossible / sensor-fault readings (the injected anomalies).
        | (df["temperature"] > 150)
        | (df["current"] > 80)
        | (df["vibration"] > 12)
        | (df["load"] > 120)
        # Old machine that has barely been serviced.
        | ((df["operating_hours"] > 15000) & (df["maintenance_count"] <= 2))
        # Several signals jointly abnormal (compounding wear).
        | ((df["temperature"] > 82) & (df["vibration"] > 3.8) & (df["load"] > 80))
    )


def generate_dataset(seed: int | None = None) -> pd.DataFrame:
    """
    Generate the full synthetic dataset as a pandas DataFrame.

    Columns:
        machine_id, machine_type, temperature, vibration, current, load,
        operating_hours, maintenance_count, ambient_temperature, humidity,
        risk_score, failure
    """
    rng = np.random.default_rng(C.RANDOM_SEED if seed is None else seed)

    # Assign machine types round-robin; ~half the fleet degrades toward failure.
    type_names = list(C.MACHINE_TYPES.keys())
    n_failing = int(round(C.NUM_MACHINES * 0.5))
    failing_flags = np.array([True] * n_failing + [False] * (C.NUM_MACHINES - n_failing))
    rng.shuffle(failing_flags)

    all_records: list[dict] = []
    for m in range(C.NUM_MACHINES):
        machine_type = type_names[m % len(type_names)]
        all_records.extend(
            _generate_machine_series(rng, m, machine_type, bool(failing_flags[m]))
        )

    df = pd.DataFrame(all_records)

    # Continuous risk score per snapshot.
    df["risk_score"] = [_compute_risk_score(r, rng) for r in df.to_dict("records")]

    # ----- Failure labels: engineering rules + risk-score calibration -----
    # Definite failures from explicit rules (always includes anomalies).
    rule_fail = _engineering_failure_rules(df)

    # Top up with the highest-risk remaining rows so the overall failure ratio
    # lands near the target (~20%). This keeps labels grounded in the risk_score
    # while never dropping a rule-based (extreme) failure.
    n_target = int(round(C.TARGET_FAILURE_RATIO * len(df)))
    n_needed = n_target - int(rule_fail.sum())

    failure = rule_fail.copy()
    if n_needed > 0:
        remaining = df.loc[~rule_fail, "risk_score"]
        # Threshold = the n_needed-th largest remaining risk score.
        threshold = remaining.nlargest(n_needed).min()
        failure = failure | ((~rule_fail) & (df["risk_score"] >= threshold))

    df["failure"] = failure.astype(int)
    df["risk_score"] = df["risk_score"].round(4)
    return df
