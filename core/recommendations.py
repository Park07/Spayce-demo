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


def build_operator_actions(zone_metrics: list[dict], zone_predictions: dict, max_capacity: int = 10) -> list[str]:
    """Generate specific operator actions based on zone status and predictions.
    These are the execution layer - telling operators WHAT TO DO, not just what's happening."""
    actions = []
    for zm in zone_metrics:
        zname = zm["zone"]
        count = zm["count"]
        ratio = count / max(1, max_capacity)

        # Current danger actions
        if ratio >= 1.0:
            actions.append(f"IMMEDIATE: Close entry to {zname} - capacity exceeded ({count}/{max_capacity})")
            actions.append(f"IMMEDIATE: Deploy security to {zname} to manage crowd flow")
        elif ratio >= 0.8:
            actions.append(f"CAUTION: {zname} approaching capacity ({count}/{max_capacity}) - prepare to restrict entry")

    # Prediction-based actions
    for zname, zpred in zone_predictions.items():
        ttc = zpred.get("time_to_critical")
        src = zpred.get("primary_inflow_source")
        trend = zpred.get("risk_trend", "stable")

        if ttc is not None and ttc < 120:
            if src:
                actions.append(f"PRE-EMPTIVE: Restrict inflow at {src} - {zname} critical in {ttc:.0f}s")
            actions.append(f"PRE-EMPTIVE: Announce crowd redistribution on PA for {zname}")
        elif trend == "rapidly_increasing":
            actions.append(f"WATCH: {zname} density rising fast - standby to restrict entry")

    if not actions:
        actions.append("ALL CLEAR: All zones within safe limits. Continue monitoring.")

    return actions
