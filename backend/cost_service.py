"""
Business-impact layer: unplanned downtime avoidance + cost impact.

Judges' asks:
  1) How the system avoids UNPLANNED downtime.
  2) How that translates into COST impact.

This is deterministic business math (not ML, not GenAI) built on transparent,
configurable assumptions -- so we can defend every number in the demo.

Idea: when the model flags a machine as High risk, the engineer can act
BEFORE it fails. Catching it early converts a costly *unplanned* breakdown into
a cheaper *planned* maintenance action. We quantify the downtime hours avoided
and the money saved.
"""

import os

# ---------------------------------------------------------------------------
# Configurable assumptions (overridable via environment variables).
# All values are illustrative industry-style figures for the demo.
# ---------------------------------------------------------------------------
COST_ASSUMPTIONS = {
    # Average hours a machine is down during an UNPLANNED failure.
    "unplanned_downtime_hours": float(os.getenv("PMS_UNPLANNED_HOURS", "8")),
    # Average hours of downtime for a PLANNED maintenance action.
    "planned_downtime_hours": float(os.getenv("PMS_PLANNED_HOURS", "2")),
    # Cost of one hour of lost production while a machine is down (INR).
    "downtime_cost_per_hour": float(os.getenv("PMS_DOWNTIME_COST_PER_HOUR", "25000")),
    # Direct repair cost of an UNPLANNED failure (parts + emergency labour, INR).
    "unplanned_repair_cost": float(os.getenv("PMS_UNPLANNED_REPAIR", "150000")),
    # Direct cost of a PLANNED maintenance action (scheduled, cheaper, INR).
    "planned_maintenance_cost": float(os.getenv("PMS_PLANNED_COST", "40000")),
    "currency": os.getenv("PMS_CURRENCY", "₹"),  # ₹
}

# How strongly each risk tier counts toward "will fail if ignored".
# Two tiers only: a High-risk machine is a failure to act on; a Low-risk one is
# not. The confidence (probability) still scales the value inside the formula.
RISK_ACTION_WEIGHT = {"High": 1.0, "Low": 0.0}


def per_machine_impact(risk_label: str, confidence: float) -> dict:
    """
    Estimate downtime avoided and cost saved for a single flagged machine.

    A flagged machine (High) that is inspected early avoids the delta
    between an unplanned failure and a planned maintenance action, scaled by how
    likely it is to actually fail (risk weight * confidence).
    """
    a = COST_ASSUMPTIONS
    weight = RISK_ACTION_WEIGHT.get(risk_label, 0.0) * float(confidence)

    downtime_avoided = (a["unplanned_downtime_hours"] - a["planned_downtime_hours"]) * weight

    unplanned_total = (
        a["unplanned_downtime_hours"] * a["downtime_cost_per_hour"] + a["unplanned_repair_cost"]
    )
    planned_total = (
        a["planned_downtime_hours"] * a["downtime_cost_per_hour"] + a["planned_maintenance_cost"]
    )
    cost_saved = (unplanned_total - planned_total) * weight

    return {
        "downtime_hours_avoided": round(downtime_avoided, 2),
        "cost_saved": round(cost_saved, 2),
    }


def fleet_impact(machines: list) -> dict:
    """
    Aggregate business impact across all predicted machines.

    `machines` is a list of dicts with at least risk_label + confidence.
    """
    total_downtime = 0.0
    total_cost = 0.0
    flagged = 0

    for m in machines:
        imp = per_machine_impact(m.get("risk_label", "Low"), m.get("confidence", 0.0))
        total_downtime += imp["downtime_hours_avoided"]
        total_cost += imp["cost_saved"]
        if m.get("risk_label") == "High":
            flagged += 1

    return {
        "machines_flagged_for_inspection": flagged,
        "unplanned_downtime_hours_avoided": round(total_downtime, 1),
        "estimated_cost_saved": round(total_cost, 2),
        "currency": COST_ASSUMPTIONS["currency"],
        "assumptions": COST_ASSUMPTIONS,
    }
