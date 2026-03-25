from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from core.config import DEFAULT_BOTTLENECK_ZONES, DEFAULT_ENTRY_ZONES, DEFAULT_EXIT_ZONES, DEFAULT_GRID_ZONES


def default_zone_templates() -> list[dict[str, Any]]:
    return DEFAULT_GRID_ZONES


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


# ─── Color system ───────────────────────────────────────────────

def _severity_color(ratio: float) -> tuple[int, int, int]:
    """BGR color ramp: green → yellow → orange → red."""
    if ratio < 0.5:
        return (0, 180, 0)
    elif ratio < 0.7:
        return (0, 200, 200)
    elif ratio < 0.85:
        return (0, 130, 255)
    else:
        return (0, 0, 220)


def _severity_word(ratio: float) -> str:
    if ratio < 0.5:
        return "SAFE"
    elif ratio < 0.7:
        return "BUSY"
    elif ratio < 0.85:
        return "CROWDED"
    else:
        return "CRITICAL"


# ─── Drawing helpers ────────────────────────────────────────────

def _pill(frame, text: str, x: int, y: int, font_scale: float,
          fg: tuple[int, int, int], thickness: int = 1,
          bg: tuple[int, int, int] = (15, 15, 15), alpha: float = 0.78,
          pad_x: int = 8, pad_y: int = 5) -> int:
    """Draw text inside a rounded-corner dark pill. Returns pill width."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    rx1 = x - pad_x
    ry1 = y - th - pad_y
    rx2 = x + tw + pad_x
    ry2 = y + bl + pad_y
    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), bg, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, font_scale, fg, thickness, cv2.LINE_AA)
    return rx2 - rx1


# ─── Zone drawing ───────────────────────────────────────────────

def draw_zones(
    frame,
    zones: list[dict[str, Any]],
    zone_metrics: list[dict[str, Any]] | None = None,
    max_capacity: int = 15,
    dominant_directions: dict[str, str] | None = None,
) -> None:
    """Operator-focused zone rendering. Big status labels, color-coded fills."""
    counts = {}
    if zone_metrics:
        for zm in zone_metrics:
            counts[zm["zone"]] = zm["count"]

    for zone in zones:
        x1, y1, x2, y2 = zone["rect"]
        name = zone["name"]
        count = counts.get(name, 0)
        ratio = count / max(1, max_capacity)
        color = _severity_color(ratio)
        status = _severity_word(ratio)

        # ── Zone fill: stronger tint when more crowded ──
        fill_alpha = min(0.30, ratio * 0.35)
        if fill_alpha > 0.02:
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(ov, fill_alpha, frame, 1 - fill_alpha, 0, frame)

        # ── Border ──
        thick = 3 if ratio >= 0.85 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        # ── Corner brackets on crowded / critical zones ──
        if ratio >= 0.7:
            L = min(18, (x2 - x1) // 6, (y2 - y1) // 6)
            for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                    (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 3)
                cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 3)

        # ── BIG status pill (centered top of zone) ──
        label = f"{name}  {count}/{max_capacity}"
        cx = (x1 + x2) // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(label, font, 0.55, 2)
        _pill(frame, label, cx - tw // 2, y1 + 22, 0.55, color, 2)

        # Status word below
        (sw, _), _ = cv2.getTextSize(status, font, 0.50, 2)
        status_color = (255, 255, 255) if ratio >= 0.85 else color
        _pill(frame, status, cx - sw // 2, y1 + 44, 0.50, status_color, 2,
              bg=color if ratio >= 0.85 else (15, 15, 15))

        # ── Direction arrow ──
        if dominant_directions and name in dominant_directions:
            d = dominant_directions[name]
            arrows = {"EAST": (30, 0), "WEST": (-30, 0),
                      "NORTH": (0, -30), "SOUTH": (0, 30)}
            if d in arrows:
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2 + 20
                adx, ady = arrows[d]
                cv2.arrowedLine(frame, (mx - adx // 2, my - ady // 2),
                                (mx + adx // 2, my + ady // 2),
                                (255, 255, 255), 3, tipLength=0.35)
                cv2.arrowedLine(frame, (mx - adx // 2, my - ady // 2),
                                (mx + adx // 2, my + ady // 2),
                                color, 2, tipLength=0.35)


# ─── Warning banner ─────────────────────────────────────────────

def draw_warning_banner(frame, danger_zones: list[str], frame_width: int,
                        zone_predictions: dict | None = None) -> int:
    """Red banner with operator action. Returns y offset."""
    if not danger_zones:
        return 0

    h = 48
    # Solid dark-red bar
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (frame_width, h), (0, 0, 160), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, h), (frame_width, h), (80, 80, 220), 2)

    # Build action text
    zn = danger_zones[0]
    # Find inflow source from predictions
    src = None
    if zone_predictions:
        zp = zone_predictions.get(zn, {})
        src = zp.get("primary_inflow_source")

    if src:
        text = f"CRITICAL: {zn} over capacity  |  Restrict entry at {src}  |  Deploy security now"
    else:
        text = f"CRITICAL: {zn} over capacity  |  Deploy security now"

    cv2.putText(frame, text, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)

    return h + 2


# ─── Status strip (minimal) ────────────────────────────────────

def draw_status_strip(frame, people_count: int, zone_metrics: list[dict],
                      max_capacity: int, frame_width: int,
                      y_start: int = 0) -> int:
    """Thin status strip: just people count + per-zone mini indicators."""
    h = 28
    y = y_start
    ov = frame.copy()
    cv2.rectangle(ov, (0, y), (frame_width, y + h), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # People count
    cv2.putText(frame, f"People: {people_count}", (10, y + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # Mini zone indicators: colored dots + name
    x_cursor = 140
    for zm in zone_metrics:
        ratio = zm["count"] / max(1, max_capacity)
        c = _severity_color(ratio)
        cv2.circle(frame, (x_cursor, y + 14), 6, c, -1)
        cv2.circle(frame, (x_cursor, y + 14), 6, (255, 255, 255), 1)
        label = f"{zm['zone']}: {zm['count']}"
        cv2.putText(frame, label, (x_cursor + 12, y + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        x_cursor += 130

    return y + h + 1


# ─── Prediction strip (only when debug/detail on) ──────────────

def draw_prediction_detail(frame, zone_predictions: dict,
                           y_start: int = 36) -> None:
    """Compact prediction info — only shown when detail mode is on."""
    y = y_start
    for zname, zp in zone_predictions.items():
        ttc = zp.get("time_to_critical")
        trend = zp.get("risk_trend", "stable")
        pred = zp.get("predicted_density", 0)

        if trend in ("rapidly_increasing",):
            c = (0, 0, 255)
        elif "increasing" in trend:
            c = (0, 165, 255)
        elif "decreasing" in trend:
            c = (200, 150, 0)
        else:
            c = (0, 200, 0)

        # Operator-friendly language
        if ttc is not None and ttc < 120:
            ttc_text = f"critical in {ttc:.0f}s"
        elif ttc is not None:
            ttc_text = f"~{ttc:.0f}s to limit"
        else:
            ttc_text = "within limits"

        trend_word = {"stable": "steady", "increasing": "rising",
                      "rapidly_increasing": "SURGING", "decreasing": "easing",
                      "rapidly_decreasing": "clearing"}.get(trend, trend)

        text = f"{zname}: {trend_word}  |  {ttc_text}"
        _pill(frame, text, 10, y, 0.38, c, 1, alpha=0.7)
        y += 22