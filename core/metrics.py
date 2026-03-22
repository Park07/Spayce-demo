from __future__ import annotations

from math import atan2, hypot
from typing import Any

import numpy as np


def compute_people_density(people_count: int, frame_area: int) -> float:
    if frame_area <= 0:
        return 0.0
    return people_count / float(frame_area)


def compute_avg_speed(tracked_people: list[dict[str, Any]]) -> float:
    speeds = []
    for person in tracked_people:
        trajectory = person.get("trajectory", [])
        if len(trajectory) < 2:
            continue
        (x1, y1), (x2, y2) = trajectory[-2], trajectory[-1]
        speeds.append(hypot(x2 - x1, y2 - y1))
    if not speeds:
        return 0.0
    return float(np.mean(speeds))


def compute_direction_consistency(tracked_people: list[dict[str, Any]]) -> float:
    angles = []
    for person in tracked_people:
        trajectory = person.get("trajectory", [])
        if len(trajectory) < 2:
            continue
        (x1, y1), (x2, y2) = trajectory[-2], trajectory[-1]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        angles.append(atan2(dy, dx))

    if len(angles) < 2:
        return 1.0
    vector_mean = abs(np.mean(np.exp(1j * np.array(angles))))
    return float(vector_mean)


def compute_clustering_pressure(tracked_people: list[dict[str, Any]], distance_threshold: float = 35.0) -> float:
    centroids = [p["centroid"] for p in tracked_people]
    if len(centroids) < 2:
        return 0.0
    close_pairs = 0
    total_pairs = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            total_pairs += 1
            if hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1]) <= distance_threshold:
                close_pairs += 1
    if total_pairs == 0:
        return 0.0
    return close_pairs / total_pairs
