from __future__ import annotations

from typing import Any


def build_alerts(
    risk_level: str,
    predicted_risk: float,
    anomalies: list[dict[str, Any]],
    zone_metrics: list[dict[str, Any]],
) -> list[str]:
    alerts: list[str] = []
    if risk_level in {"HIGH", "CRITICAL"}:
        alerts.append(f"Crowd Stress Risk currently {risk_level}.")
    if predicted_risk >= 0.65:
        alerts.append("Predicted congestion escalation in next 15 seconds.")

    for anomaly in anomalies[-3:]:
        zone = anomaly.get("zone", "unknown zone")
        if anomaly["type"] == "stopped_exit_flow":
            alerts.append(f"Flow disruption detected near {zone}.")
        elif anomaly["type"] == "bottleneck_cluster":
            alerts.append(f"Crowd Stress Risk rising near {zone}.")
        else:
            alerts.append(f"{anomaly['explanation']} ({zone})")

    if zone_metrics:
        top_zone = max(zone_metrics, key=lambda z: z["local_risk"])
        if top_zone["local_risk"] > 0.65:
            alerts.append(f"High local pressure in {top_zone['zone']}.")

    deduped = []
    for a in alerts:
        if a not in deduped:
            deduped.append(a)
    return deduped


def build_recommendations(alerts: list[str], zone_metrics: list[dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    for alert in alerts:
        if "Exit" in alert:
            recommendations.append("Deploy staff to Exit A and clear blocked lane.")
        if "bottleneck" in alert.lower() or "pressure" in alert.lower():
            recommendations.append("Open alternate route near bottleneck zone.")
        if "escalation" in alert.lower():
            recommendations.append("Pre-position response team for next 15 minutes.")
        if "Flow disruption" in alert:
            recommendations.append("Investigate slowed movement and broadcast directional guidance.")

    if zone_metrics:
        densest = max(zone_metrics, key=lambda z: z["local_density"])
        recommendations.append(f"Monitor {densest['zone']} continuously for local surges.")

    if not recommendations:
        recommendations.append("Continue monitoring; maintain current staffing posture.")

    ordered = []
    for item in recommendations:
        if item not in ordered:
            ordered.append(item)
    return ordered
