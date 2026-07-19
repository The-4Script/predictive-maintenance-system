"""
GenAI explanation layer (Gemini) with engineering-grade reasoning.

Responsibility: turn an ML prediction into a professional predictive-maintenance
report. The layer distinguishes three situations before recommending anything:

  1. SENSOR ANOMALY  -- physically impossible readings (e.g. 5200 C). The data,
     not the machine, is most likely at fault. Recommend verifying the sensor
     BEFORE any expensive maintenance.
  2. CRITICAL / DEGRADATION -- readings are high but physically plausible, so the
     machine genuinely needs inspection/maintenance.
  3. NORMAL -- everything within safe ranges; routine monitoring.

Guardrails (unchanged): scope-locked prompt, recommend-only, disclaimer always
attached, safe deterministic fallback when the API key is missing or a call
fails. Token use stays small: we send one machine's sensor values + the detected
anomalies, never the database.

NOTE: this module does NOT change the ML prediction, risk score, or confidence.
It only improves the explanation/recommendation text.
"""

import os
from typing import List, Optional

DISCLAIMER = (
    "This tool provides decision support only. "
    "Final approval must be made by an authorized person."
)

MODEL_NAME = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
MAX_OUTPUT_TOKENS = int(os.getenv("GENAI_MAX_TOKENS", "600"))

# ---------------------------------------------------------------------------
# Engineering reference ranges (generic, machine-agnostic for the MVP).
# For each parameter:
#   safe   = normal healthy operating band
#   impossible = physically implausible => points to a SENSOR fault, not wear
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "temperature": {
        "unit": "C", "safe": (30, 85), "impossible": (-20, 200),
        "causes_high": "bearing friction, cooling-system failure, or overload",
    },
    "vibration": {
        "unit": "mm/s", "safe": (0, 4.5), "impossible": (None, 15),
        "causes_high": "bearing wear, shaft misalignment, or imbalance",
    },
    "current": {
        "unit": "A", "safe": (0, 32), "impossible": (None, 100),
        "causes_high": "mechanical resistance, winding fault, or overload",
    },
    "load": {
        "unit": "%", "safe": (0, 90), "impossible": (0, 120),
        "causes_high": "sustained over-loading beyond rated capacity",
    },
    "humidity": {
        "unit": "%", "safe": (0, 100), "impossible": (0, 100),
        "causes_high": "environmental moisture / corrosion risk",
    },
    "ambient_temperature": {
        "unit": "C", "safe": (5, 55), "impossible": (-30, 80),
        "causes_high": "poor ventilation or hot environment",
    },
}

SENSOR_FAULT_CAUSES = [
    "faulty or failing sensor",
    "calibration drift",
    "loose wiring / bad connection",
    "data-acquisition or communication error",
    "corrupted sensor reading",
    "(less likely) an extremely severe machine failure",
]

# Normal-case predictive-maintenance actions (when values are realistic).
NORMAL_ACTIONS = [
    "Inspect bearings for wear",
    "Lubricate moving components",
    "Check shaft alignment",
    "Inspect the cooling system",
    "Consider reducing operating load",
    "Schedule preventive maintenance",
]


def is_genai_enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# Sensor analysis (pure Python -- deterministic, no LLM needed)
# ---------------------------------------------------------------------------
def _classify_parameter(name: str, value: float) -> Optional[dict]:
    """
    Classify one sensor reading as 'impossible' (sensor fault) or 'abnormal'
    (real high reading) or None (within safe range).
    """
    spec = PARAM_RANGES.get(name)
    if spec is None or value is None:
        return None

    lo_imp, hi_imp = spec["impossible"]
    lo_safe, hi_safe = spec["safe"]

    impossible = (hi_imp is not None and value > hi_imp) or (
        lo_imp is not None and value < lo_imp
    )
    abnormal = (hi_safe is not None and value > hi_safe) or (
        lo_safe is not None and value < lo_safe
    )

    if not impossible and not abnormal:
        return None

    return {
        "name": name,
        "value": value,
        "unit": spec["unit"],
        "safe_range": f"{lo_safe}-{hi_safe} {spec['unit']}",
        "impossible": impossible,
        "causes_high": spec["causes_high"],
    }


def _analyse_sensors(sensors: Optional[dict], risk_label: str) -> dict:
    """
    Inspect all provided sensor values and decide the overall situation:
      * 'sensor_anomaly' -- any physically impossible reading (data/sensor fault),
      * 'degradation'    -- the model predicts High risk from plausible readings,
      * 'normal'         -- Low risk and nothing impossible.

    The mode respects the ML risk label: a High-risk machine is never reported as
    'normal', even if individual readings sit just under the safe caps.
    """
    findings: List[dict] = []
    if sensors:
        for name, value in sensors.items():
            try:
                f = _classify_parameter(name, float(value))
            except (TypeError, ValueError):
                f = None
            if f:
                findings.append(f)

    has_impossible = any(f["impossible"] for f in findings)
    if has_impossible:
        mode = "sensor_anomaly"
    elif risk_label == "High":
        mode = "degradation"
    else:
        mode = "normal"
    return {"mode": mode, "findings": findings}


# ---------------------------------------------------------------------------
# Deterministic report builder (fallback + grounding for the LLM)
# Uses light HTML (<br>, <b>) so the existing frontend renders it as a report.
# ---------------------------------------------------------------------------
def _build_report(
    machine_id: str, risk_label: str, confidence: float, analysis: dict,
    top_features: Optional[List[dict]] = None,
) -> str:
    mode = analysis["mode"]
    findings = analysis["findings"]
    conf = f"{confidence:.0%}"

    head = (
        f"<b>Machine {machine_id}</b><br>"
        f"<b>Risk:</b> {risk_label} &nbsp; <b>Confidence:</b> {conf}<br><br>"
    )

    if mode == "sensor_anomaly":
        summary = (
            "<b>Summary:</b> One or more readings are <b>physically impossible</b>. "
            "The data most likely indicates a SENSOR problem rather than the machine "
            "itself. Verify the sensor before authorising expensive maintenance.<br><br>"
        )
        params = "<b>Abnormal parameters</b><br>"
        for f in findings:
            tag = " (physically impossible)" if f["impossible"] else ""
            params += (
                f"• <b>{f['name'].replace('_', ' ').title()}</b>{tag}<br>"
                f"&nbsp;&nbsp;Observed: {f['value']} {f['unit']}<br>"
                f"&nbsp;&nbsp;Expected safe range: {f['safe_range']}<br>"
                f"&nbsp;&nbsp;Analysis: this value is outside any realistic operating "
                f"range, so it most likely reflects a data problem.<br>"
            )
        causes = "<br><b>Possible causes:</b> " + "; ".join(SENSOR_FAULT_CAUSES) + ".<br>"
        actions = (
            "<br><b>Priority actions</b><br>"
            "1. Verify sensor readings — inspect calibration, check wiring, repeat the measurement.<br>"
            "2. Inspect the affected machine subsystem only if the reading is confirmed.<br>"
            "3. Schedule maintenance <i>only</i> after abnormal readings are validated.<br>"
        )
        return head + summary + params + causes + actions

    if mode == "degradation":
        summary = (
            f"<b>Summary:</b> The model predicts <b>{risk_label}</b> risk from "
            "genuinely elevated but physically plausible readings, consistent with "
            "machine wear/degradation.<br><br>"
        )
        if findings:
            params = "<b>Abnormal parameters</b><br>"
            for f in findings:
                params += (
                    f"• <b>{f['name'].replace('_', ' ').title()}</b><br>"
                    f"&nbsp;&nbsp;Observed: {f['value']} {f['unit']}<br>"
                    f"&nbsp;&nbsp;Expected safe range: {f['safe_range']}<br>"
                    f"&nbsp;&nbsp;Analysis: above the safe band — likely {f['causes_high']}.<br>"
                )
        else:
            # Readings are just under the safe caps but the model still flags High
            # risk -- point to the top contributing signals.
            factors = ", ".join(f["feature"] for f in (top_features or [])) or "the monitored signals"
            params = (
                "<b>Key risk factors</b><br>"
                f"• The model's decision is driven mainly by <b>{factors}</b>, "
                "trending toward the abnormal range.<br>"
            )
        actions = (
            "<br><b>Priority actions</b><br>"
            "1. Inspect the components tied to the abnormal signals.<br>"
            "2. Lubricate / realign as needed and re-check.<br>"
            "3. Schedule preventive maintenance before continued operation.<br>"
        )
        return head + summary + params + actions

    # normal
    summary = (
        "<b>Summary:</b> All readings are within safe ranges and the failure risk "
        "is low. No sensor anomaly detected.<br><br>"
    )
    actions = "<b>Routine recommendations</b><br>" + "".join(
        f"• {a}<br>" for a in NORMAL_ACTIONS[:4]
    )
    return head + summary + actions


def _build_llm_prompt(
    machine_id: str, risk_label: str, confidence: float, analysis: dict, sensors: dict
) -> str:
    """Grounded prompt: give the model the facts + detected anomalies + format."""
    mode = analysis["mode"]
    readings = ", ".join(f"{k}={v}" for k, v in (sensors or {}).items()) or "n/a"
    flagged = "; ".join(
        f"{f['name']}={f['value']}{f['unit']} (safe {f['safe_range']}"
        + (", PHYSICALLY IMPOSSIBLE" if f["impossible"] else "") + ")"
        for f in analysis["findings"]
    ) or "none"

    guidance = {
        "sensor_anomaly": (
            "At least one reading is physically impossible. Treat this as a likely "
            "SENSOR / data fault, not machine overheating. Recommend verifying the "
            "sensor (calibration, wiring, repeat measurement) BEFORE any expensive "
            "maintenance."
        ),
        "degradation": (
            "Readings are high but plausible, so the machine genuinely needs "
            "inspection/maintenance. Do NOT mention sensor failure."
        ),
        "normal": (
            "Readings are within safe ranges. Give routine preventive-maintenance "
            "advice. Do NOT mention sensor failure."
        ),
    }[mode]

    return (
        f"Machine: {machine_id}\n"
        f"Model risk: {risk_label} (confidence {confidence:.0%})\n"
        f"Sensor readings: {readings}\n"
        f"Flagged parameters: {flagged}\n"
        f"Situation: {guidance}\n\n"
        "Write a concise professional predictive-maintenance report with these "
        "sections: Summary; Abnormal Parameters (for each: observed value, expected "
        "safe range, why it is abnormal, possible engineering causes); Priority "
        "Actions (numbered). Explain WHY each parameter is abnormal and WHY the risk "
        "is what it is. Use short lines separated by newlines."
    )


SYSTEM_PROMPT = (
    "You are an industrial predictive-maintenance engineer. Only discuss machine "
    "health, sensor diagnostics, and inspection/maintenance actions. Distinguish "
    "impossible sensor readings (sensor/data faults) from genuine machine wear. "
    "Recommend only -- never issue a final repair/replace decision. Be precise and "
    "concise (max ~200 words)."
)


def generate_explanation(
    machine_id: str,
    risk_label: str,
    confidence: float,
    top_features: List[dict],
    sensors: Optional[dict] = None,
) -> dict:
    """
    Return an engineering-grade explanation + prioritized recommendation.

    `sensors` is an optional dict of the machine's sensor readings; when present
    it enables sensor-anomaly detection. Falls back to a deterministic report if
    the API key is missing or the LLM call fails.
    """
    analysis = _analyse_sensors(sensors, risk_label)
    report = _build_report(machine_id, risk_label, confidence, analysis, top_features)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "machine_id": machine_id,
            "explanation": report,
            "source": "fallback-template",
            "disclaimer": DISCLAIMER,
        }

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
        prompt = _build_llm_prompt(machine_id, risk_label, confidence, analysis, sensors or {})
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.3},
        )
        text = (response.text or "").strip()
        if not text:
            text = report
        else:
            # Render newlines as line breaks for the existing frontend div.
            text = text.replace("\n", "<br>")
        return {
            "machine_id": machine_id,
            "explanation": text,
            "source": MODEL_NAME,
            "disclaimer": DISCLAIMER,
        }
    except Exception:
        return {
            "machine_id": machine_id,
            "explanation": report,
            "source": "fallback-template",
            "disclaimer": DISCLAIMER,
        }
