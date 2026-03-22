from __future__ import annotations

from typing import Any

import cv2

from core.config import DEFAULT_BOTTLENECK_ZONES, DEFAULT_ENTRY_ZONES, DEFAULT_EXIT_ZONES


def default_zone_templates() -> list[dict[str, Any]]:
    return [*DEFAULT_EXIT_ZONES, *DEFAULT_ENTRY_ZONES, *DEFAULT_BOTTLENECK_ZONES]


def build_zones_from_normalized(normalized_zones: list[dict[str, Any]], frame_width: int, frame_height: int) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    for z in normalized_zones:
        x1n, y1n, x2n, y2n = z["rect"]
        zones.append(
            {
                "name": z["name"],
                "type": z["type"],
                "rect": (
                    int(max(0.0, min(1.0, x1n)) * frame_width),
                    int(max(0.0, min(1.0, y1n)) * frame_height),
                    int(max(0.0, min(1.0, x2n)) * frame_width),
                    int(max(0.0, min(1.0, y2n)) * frame_height),
                ),
            }
        )
    return zones


def build_default_zones(frame_width: int, frame_height: int) -> list[dict[str, Any]]:
    return build_zones_from_normalized(default_zone_templates(), frame_width, frame_height)


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_zone_metrics(
    zones: list[dict[str, Any]], tracked_people: list[dict[str, Any]], frame_area: int
) -> list[dict[str, Any]]:
    zone_data: list[dict[str, Any]] = []
    for zone in zones:
        x1, y1, x2, y2 = zone["rect"]
        area = max(1, (x2 - x1) * (y2 - y1))
        in_zone = [p for p in tracked_people if point_in_rect(p["centroid"], zone["rect"])]
        count = len(in_zone)
        local_density = count / float(area)
        local_risk = min(1.0, (local_density * frame_area) / 18.0)
        zone_data.append(
            {
                "zone": zone["name"],
                "zone_type": zone["type"],
                "count": count,
                "local_density": local_density,
                "local_risk": local_risk,
            }
        )
    return zone_data


def draw_zones(frame, zones: list[dict[str, Any]]) -> None:
    color_map = {"exit": (0, 200, 255), "entry": (0, 255, 100), "bottleneck": (120, 120, 255)}
    for zone in zones:
        x1, y1, x2, y2 = zone["rect"]
        color = color_map.get(zone["type"], (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, zone["name"], (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
