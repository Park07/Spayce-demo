from __future__ import annotations

APP_TITLE = "Crowd Stress Detector"
VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv"]

PERSON_CLASS_ID = 0
YOLO_MODEL_NAME = "yolov8x.pt"
TRACKER_CONFIG = "bytetrack.yaml"
TRACK_MAX_DETECTIONS = 300

DEFAULT_EXIT_ZONES = [
    {"name": "Exit A", "type": "exit", "rect": (0.65, 0.0, 1.0, 1.0)},
]
DEFAULT_ENTRY_ZONES = [
    {"name": "Gate B", "type": "entry", "rect": (0.0, 0.0, 0.35, 1.0)},
]
DEFAULT_BOTTLENECK_ZONES = [
    {"name": "Bottleneck C", "type": "bottleneck", "rect": (0.20, 0.0, 0.75, 1.0)},
]

HEATMAP_BLUR_KERNEL = (41, 41)

RISK_WEIGHTS = {
    "density": 0.25,
    "density_growth": 0.16,
    "zone_congestion": 0.18,
    "movement_slowdown": 0.15,
    "direction_conflict": 0.12,
    "bottleneck_pressure": 0.08,
    "anomaly_pressure": 0.06,
}

RISK_LEVELS = [
    (0.25, "LOW"),
    (0.50, "MEDIUM"),
    (0.75, "HIGH"),
    (1.01, "CRITICAL"),
]

ANOMALY_THRESHOLDS = {
    "density_spike_delta": 0.12,
    "direction_conflict_high": 0.55,
    "exit_speed_low": 2.5,
    "bottleneck_density_high": 0.08,
}

PREDICTION_WINDOW_SECONDS = 15
MAX_REASON_COUNT = 4
# Zone adjacency (directional flow graph)
# Weight: 1.0 = directly connected, exp(-1) ≈ 0.37 = 2 hops, from BINTS k-hop (Section 4.1)
ZONE_ADJACENCY = {
    ("Gate B", "Bottleneck C"): 1.0,
    ("Bottleneck C", "Exit A"): 1.0,
    ("Gate B", "Exit A"): 0.37,
}

# Research-backed crowd density thresholds (persons/m²)
# Source: Itaewon analysis (ScienceDirect 2025), Keith Still PhD, PLOS ONE 2021
CROWD_DENSITY_THRESHOLDS = {
    "free_flow": 2.0,
    "maximum_flow": 4.0,
    "critical_stationary": 4.7,
    "critical_moving": 4.0,
    "crush_risk": 5.0,
    "itaewon_critical": 6.9,
}

# BINTS-inspired prediction weights (from Table 5, Section 5.5.2)
PREDICTION_WEIGHTS = {
    "prediction": 0.5,
    "spatial": 0.25,
    "temporal": 0.25,
}