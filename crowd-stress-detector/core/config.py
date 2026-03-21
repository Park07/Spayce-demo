from __future__ import annotations

APP_TITLE = "Crowd Stress Detector"
VIDEO_EXTENSIONS = ["mp4", "mov", "avi", "mkv"]

PERSON_CLASS_ID = 0
YOLO_MODEL_NAME = "yolov8m.pt"
TRACKER_CONFIG = "bytetrack.yaml"
TRACK_MAX_DETECTIONS = 300

DEFAULT_EXIT_ZONES = [
    {"name": "Exit A", "type": "exit", "rect": (0.78, 0.65, 0.98, 0.98)},
]
DEFAULT_ENTRY_ZONES = [
    {"name": "Gate B", "type": "entry", "rect": (0.02, 0.60, 0.20, 0.98)},
]
DEFAULT_BOTTLENECK_ZONES = [
    {"name": "Bottleneck C", "type": "bottleneck", "rect": (0.40, 0.35, 0.62, 0.80)},
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
