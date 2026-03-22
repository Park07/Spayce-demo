from __future__ import annotations

from typing import Any

from core.config import ANOMALY_THRESHOLDS


def detect_anomalies(
    frame_idx: int,
    density: float,
    previous_density: float,
    directional_consistency: float,
    avg_speed: float,
    zone_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    if density - previous_density > ANOMALY_THRESHOLDS["density_spike_delta"]:
        anomalies.append(
            {
                "frame": frame_idx,
                "severity": "HIGH",
                "zone": "Global",
                "explanation": "Sudden density spike detected.",
                "type": "density_spike",
            }
        )

    if (1.0 - directional_consistency) > ANOMALY_THRESHOLDS["direction_conflict_high"]:
        anomalies.append(
            {
                "frame": frame_idx,
                "severity": "MEDIUM",
                "zone": "Global",
                "explanation": "Abnormal movement direction conflict detected.",
                "type": "direction_conflict",
            }
        )

    for zone in zone_metrics:
        if zone["zone_type"] == "exit" and avg_speed < ANOMALY_THRESHOLDS["exit_speed_low"] and zone["count"] > 2:
            anomalies.append(
                {
                    "frame": frame_idx,
                    "severity": "HIGH",
                    "zone": zone["zone"],
                    "explanation": "Stopped flow near exit zone.",
                    "type": "stopped_exit_flow",
                }
            )
        if (
            zone["zone_type"] == "bottleneck"
            and zone["local_density"] > ANOMALY_THRESHOLDS["bottleneck_density_high"]
            and zone["count"] >= 3
        ):
            anomalies.append(
                {
                    "frame": frame_idx,
                    "severity": "HIGH",
                    "zone": zone["zone"],
                    "explanation": "Clustering near bottleneck.",
                    "type": "bottleneck_cluster",
                }
            )
    return anomalies
