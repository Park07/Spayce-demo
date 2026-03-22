from __future__ import annotations

from typing import Any

import numpy as np

from core.config import MAX_REASON_COUNT, RISK_LEVELS, RISK_WEIGHTS
from core.utils import clamp_01, risk_level_from_score


def compute_crowd_stress_risk(
    people_density: float,
    density_growth: float,
    zone_congestion: float,
    avg_speed: float,
    directional_consistency: float,
    bottleneck_pressure: float,
    anomaly_count: int,
) -> dict[str, Any]:
    density_term = clamp_01(people_density * 420.0)
    growth_term = clamp_01((density_growth + 0.03) * 8.0)
    zone_term = clamp_01(zone_congestion)
    slowdown_term = clamp_01(1.0 - min(1.0, avg_speed / 12.0))
    direction_conflict = clamp_01(1.0 - directional_consistency)
    bottleneck_term = clamp_01(bottleneck_pressure * 2.5)
    anomaly_term = clamp_01(anomaly_count / 6.0)

    raw_score = (
        RISK_WEIGHTS["density"] * density_term
        + RISK_WEIGHTS["density_growth"] * growth_term
        + RISK_WEIGHTS["zone_congestion"] * zone_term
        + RISK_WEIGHTS["movement_slowdown"] * slowdown_term
        + RISK_WEIGHTS["direction_conflict"] * direction_conflict
        + RISK_WEIGHTS["bottleneck_pressure"] * bottleneck_term
        + RISK_WEIGHTS["anomaly_pressure"] * anomaly_term
    )
    current_risk = clamp_01(raw_score)
    risk_level = risk_level_from_score(current_risk, RISK_LEVELS)

    reason_map = {
        "High overall density": density_term,
        "Density rising quickly": growth_term,
        "Zone congestion elevated": zone_term,
        "Movement slowdown detected": slowdown_term,
        "Directional conflict observed": direction_conflict,
        "Bottleneck pressure increasing": bottleneck_term,
        "Multiple anomalies active": anomaly_term,
    }
    reasons = [k for k, _ in sorted(reason_map.items(), key=lambda x: x[1], reverse=True)[:MAX_REASON_COUNT]]
    return {
        "current_risk_score": current_risk,
        "risk_level": risk_level,
        "reasons": reasons,
    }


def predict_crowd_stress_risk(recent_scores: list[float], horizon_seconds: int = 15) -> float:
    if not recent_scores:
        return 0.0
    if len(recent_scores) == 1:
        return recent_scores[-1]
    x = np.arange(len(recent_scores))
    y = np.array(recent_scores)
    slope = np.polyfit(x, y, 1)[0]
    prediction = recent_scores[-1] + slope * max(3, horizon_seconds // 5)
    return clamp_01(float(prediction))
