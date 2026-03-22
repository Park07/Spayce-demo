from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlink

from core.anomaly import detect_anomalies
from core.detector import PersonDetector, draw_detections
from core.heatmap import build_heatmap_overlay, overlay_heatmap
from core.metrics import compute_avg_speed, compute_clustering_pressure, compute_direction_consistency, compute_people_density
from core.prediction import BiModalPredictor
from core.recommendations import build_alerts, build_recommendations
from core.risk import compute_crowd_stress_risk, predict_crowd_stress_risk
from core.utils import ensure_dir
from core.zones import build_default_zones, build_zones_from_normalized, compute_zone_metrics, draw_zones, draw_warning_banner, draw_status_strip, draw_prediction_detail


def _open_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, max(1.0, fps), frame_size)


def process_video_file(
    input_path: Path,
    root_dir: Path,
    show_heatmap: bool = True,
    enable_zones: bool = True,
    show_trends: bool = True,
    enable_multi_camera: bool = False,
    confidence: float = 0.35,
    sample_stride: int = 1,
    show_boxes: bool = False,
    max_capacity: int = 10,
    frame_callback: Any | None = None,
    normalized_zones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detector = PersonDetector()
    track_history: dict[int, list[tuple[int, int]]] = {}

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to read video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    frame_area = max(1, frame_width * frame_height)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    zones = []
    if enable_zones:
        if normalized_zones:
            zones = build_zones_from_normalized(normalized_zones, frame_width, frame_height)
        else:
            zones = build_default_zones(frame_width, frame_height)

    # Initialize BINTS-inspired bi-modal predictor
    zone_names = [z["name"] for z in zones] if zones else []
    zone_types = {z["name"]: z["type"] for z in zones} if zones else {}
    bimodal_predictor = BiModalPredictor(zone_names=zone_names, zone_types=zone_types)

    processed_dir = ensure_dir(root_dir / "outputs" / "processed")
    session_id = uuid.uuid4().hex[:8]
    out_path = processed_dir / f"{input_path.stem}_{session_id}_processed.mp4"
    writer = _open_video_writer(out_path, fps, frame_size)

    timeline_rows: list[dict[str, Any]] = []
    all_anomalies: list[dict[str, Any]] = []
    latest_alerts: list[str] = []
    latest_recommendations: list[str] = []
    latest_zone_metrics: list[dict[str, Any]] = []
    latest_zone_predictions: dict[str, Any] = {}
    peak_zone_metrics: list[dict[str, Any]] = []
    peak_zone_total_count = 0
    peak_zone_predictions: dict[str, Any] = {}
    previous_density = 0.0
    risk_history: list[float] = []
    bottleneck_risk_history: list[float] = []
    flow_state = _init_flow_state()
    tracking_quality = _init_tracking_quality()

    frame_idx = 0
    sahi_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if sample_stride > 1 and frame_idx % sample_stride != 0:
            writer.write(frame)
            continue

        tracked_people = detector.track_people(frame, confidence=confidence)
        if frame_idx % 5 == 0 or frame_idx == 1:
            sahi_count = len(detector.detect_people(frame, confidence=confidence))
        _update_track_history(track_history, tracked_people)
        _update_tracking_quality(tracking_quality, tracked_people)
        direction_counts = _direction_distribution(tracked_people)
        entry_exit = _update_flow_counters(tracked_people, frame_width, frame_height, flow_state)
        centroids = [p["centroid"] for p in tracked_people]
        people_count = max(len(tracked_people), sahi_count)
        density = compute_people_density(people_count, frame_area)

        avg_speed = compute_avg_speed(tracked_people)
        dir_consistency = compute_direction_consistency(tracked_people)
        cluster_pressure = compute_clustering_pressure(tracked_people)

        zone_metrics = compute_zone_metrics(zones, tracked_people, frame_area) if zones else []

        # Bi-modal prediction (BINTS-inspired)
        if zones:
            bimodal_predictor.update(tracked_people, zones)
            zone_predictions = bimodal_predictor.predict()
        else:
            zone_predictions = {}
        latest_zone_predictions = zone_predictions

        # Track peak zone metrics (busiest frame, not last frame)
        current_zone_total = sum(z["count"] for z in zone_metrics) if zone_metrics else 0
        if current_zone_total > peak_zone_total_count:
            peak_zone_total_count = current_zone_total
            peak_zone_metrics = [dict(z) for z in zone_metrics]
            peak_zone_predictions = dict(zone_predictions)

        max_zone_risk = max((z["local_risk"] for z in zone_metrics), default=0.0)
        max_zone_density = max((z["local_density"] for z in zone_metrics), default=0.0)
        bottleneck_risk = max((z["local_risk"] for z in zone_metrics if z["zone_type"] == "bottleneck"), default=0.0)
        bottleneck_risk_history.append(bottleneck_risk)
        predicted_bottleneck_risk = predict_crowd_stress_risk(bottleneck_risk_history[-20:], horizon_seconds=15)

        anomalies = detect_anomalies(
            frame_idx=frame_idx,
            density=density,
            previous_density=previous_density,
            directional_consistency=dir_consistency,
            avg_speed=avg_speed,
            zone_metrics=zone_metrics,
        )
        all_anomalies.extend(anomalies)

        risk_obj = compute_crowd_stress_risk(
            people_density=density,
            density_growth=density - previous_density,
            zone_congestion=max_zone_risk,
            avg_speed=avg_speed,
            directional_consistency=dir_consistency,
            bottleneck_pressure=cluster_pressure,
            anomaly_count=len(anomalies),
        )
        current_risk = risk_obj["current_risk_score"]
        risk_history.append(current_risk)
        predicted_risk = predict_crowd_stress_risk(risk_history[-15:], horizon_seconds=15)

        latest_alerts = build_alerts(risk_obj["risk_level"], predicted_risk, anomalies, zone_metrics)
        if predicted_bottleneck_risk >= 0.65:
            latest_alerts.append("Predicted bottleneck escalation in next 15 seconds.")

        # Add bi-modal prediction alerts
        for zname, zpred in zone_predictions.items():
            if zpred.get("time_to_critical") is not None and zpred["time_to_critical"] < 60:
                latest_alerts.append(
                    f"BIMODAL: {zname} predicted critical in {zpred['time_to_critical']:.0f}s"
                    f" (inflow from {zpred.get('primary_inflow_source', 'unknown')})"
                )
            if zpred.get("cross_modal_alert"):
                latest_alerts.append(f"BIMODAL: Unusual density-flow divergence at {zname}")
            if zpred.get("risk_trend") == "rapidly_increasing":
                latest_alerts.append(f"BIMODAL: {zname} density rapidly increasing")

        latest_recommendations = build_recommendations(latest_alerts, zone_metrics)
        latest_zone_metrics = zone_metrics

        annotated = draw_detections(frame, tracked_people) if show_boxes else frame.copy()
        if show_heatmap and centroids:
            heat = build_heatmap_overlay(frame.shape, centroids)
            annotated = overlay_heatmap(annotated, heat)

        # Compute per-zone direction for display
        zone_directions = {}
        for zone in zones:
            zname = zone["name"]
            x1z, y1z, x2z, y2z = zone["rect"]
            zone_people = [p for p in tracked_people if x1z <= p["centroid"][0] <= x2z and y1z <= p["centroid"][1] <= y2z]
            dir_counts = {}
            for p in zone_people:
                traj = p.get("trajectory", [])
                if len(traj) >= 2:
                    dx = traj[-1][0] - traj[-2][0]
                    dy = traj[-1][1] - traj[-2][1]
                    if abs(dx) < 2 and abs(dy) < 2:
                        d = "STILL"
                    elif abs(dx) >= abs(dy):
                        d = "EAST" if dx > 0 else "WEST"
                    else:
                        d = "SOUTH" if dy > 0 else "NORTH"
                    dir_counts[d] = dir_counts.get(d, 0) + 1
            if dir_counts:
                zone_directions[zname] = max(dir_counts, key=dir_counts.get)

        if zones:
            draw_zones(annotated, zones, zone_metrics=zone_metrics, max_capacity=max_capacity, dominant_directions=zone_directions)

        # --- Operator HUD ---
        danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= max_capacity * 0.8]
        y_cursor = draw_warning_banner(annotated, danger_zones, frame_width, zone_predictions=zone_predictions)
        y_cursor = draw_status_strip(annotated, people_count, zone_metrics, max_capacity, frame_width, y_start=y_cursor)
        if show_boxes:  # Detail mode: show prediction info
            draw_prediction_detail(annotated, zone_predictions, y_start=y_cursor)

        if enable_multi_camera:
            resized = cv2.resize(annotated, (frame_width // 2, frame_height // 2))
            top = np.hstack([resized, resized])
            bottom = np.hstack([resized, resized])
            annotated = cv2.resize(np.vstack([top, bottom]), frame_size)

        writer.write(annotated)
        if frame_callback is not None:
            frame_callback(
                annotated,
                frame_idx,
                total_frames,
                current_risk,
                predicted_risk,
                people_count,
            )
        timeline_rows.append(
            {
                "frame": frame_idx,
                "second": frame_idx / max(1.0, fps),
                "people_count": people_count,
                "density": density,
                "avg_speed": avg_speed,
                "directional_consistency": dir_consistency,
                "current_risk": current_risk,
                "predicted_risk": predicted_risk,
                "max_zone_density": max_zone_density,
                "predicted_bottleneck_risk": predicted_bottleneck_risk,
                "entries_total": entry_exit["entries_total"],
                "exits_total": entry_exit["exits_total"],
                "dominant_direction": _dominant_direction(direction_counts),
                "active_tracks": tracking_quality["active_tracks"],
                "id_switches": tracking_quality["id_switches"],
                "lost_tracks": tracking_quality["lost_tracks"],
                "recovered_tracks": tracking_quality["recovered_tracks"],
                "tracking_stability": _tracking_stability(tracking_quality),
                "bimodal_predicted_risk": max(
                    (zp.get("predicted_density", 0) for zp in zone_predictions.values()),
                    default=0.0,
                ) if zone_predictions else 0.0,
            }
        )
        previous_density = density

    cap.release()
    writer.release()

    timeline_df = pd.DataFrame(timeline_rows)
    if timeline_df.empty:
        timeline_df = pd.DataFrame(
            [
                {
                    "frame": 0,
                    "second": 0,
                    "people_count": 0,
                    "density": 0,
                    "avg_speed": 0,
                    "current_risk": 0,
                    "predicted_risk": 0,
                    "max_zone_density": 0,
                    "predicted_bottleneck_risk": 0,
                    "entries_total": 0,
                    "exits_total": 0,
                    "dominant_direction": "UNKNOWN",
                    "active_tracks": 0,
                    "id_switches": 0,
                    "lost_tracks": 0,
                    "recovered_tracks": 0,
                    "tracking_stability": 1.0,
                    "bimodal_predicted_risk": 0.0,
                }
            ]
        )

    busiest_zone = "N/A"
    if latest_zone_metrics:
        busiest_zone = max(latest_zone_metrics, key=lambda z: z["count"])["zone"]

    summary = {
        "session_id": session_id,
        "video_name": input_path.name,
        "peak_people_count": float(timeline_df["people_count"].max()),
        "latest_current_risk": float(timeline_df["current_risk"].iloc[-1]),
        "latest_predicted_risk": float(timeline_df["predicted_risk"].iloc[-1]),
        "highest_current_risk": float(timeline_df["current_risk"].max()),
        "highest_predicted_risk": float(timeline_df["predicted_risk"].max()),
        "risk_level": _risk_label(float(timeline_df["current_risk"].iloc[-1])),
        "busiest_zone": busiest_zone,
        "predicted_bottleneck_risk": float(timeline_df["predicted_bottleneck_risk"].iloc[-1]),
        "entries_total": int(timeline_df["entries_total"].iloc[-1]),
        "exits_total": int(timeline_df["exits_total"].iloc[-1]),
        "dominant_direction": str(timeline_df["dominant_direction"].iloc[-1]),
        "active_tracks": int(timeline_df["active_tracks"].iloc[-1]),
        "id_switches": int(timeline_df["id_switches"].iloc[-1]),
        "lost_tracks": int(timeline_df["lost_tracks"].iloc[-1]),
        "recovered_tracks": int(timeline_df["recovered_tracks"].iloc[-1]),
        "tracking_stability": float(timeline_df["tracking_stability"].iloc[-1]),
        "zone_predictions": peak_zone_predictions if peak_zone_predictions else latest_zone_predictions,
        "flow_matrix": bimodal_predictor.get_flow_matrix() if zones else {},
    }
    return {
        "summary": summary,
        "alerts": latest_alerts,
        "recommendations": latest_recommendations,
        "zone_snapshots": latest_zone_metrics,
        "peak_zone_snapshots": peak_zone_metrics if peak_zone_metrics else latest_zone_metrics,
        "timeline_df": timeline_df if show_trends else pd.DataFrame(),
        "anomalies": all_anomalies,
        "processed_video_path": str(out_path),
    }


def resolve_stream_url(url_or_stream: str) -> str:
    """Resolve webpage livestream URLs (e.g. Skyline) to direct stream URL."""
    lowered = url_or_stream.lower()
    if lowered.endswith(".m3u8") or lowered.endswith(".mp4"):
        return url_or_stream
    try:
        session = streamlink.Streamlink()
        streams = session.streams(url_or_stream)
        if not streams:
            return url_or_stream
        if "best" in streams:
            return streams["best"].url
        first = next(iter(streams.values()))
        return first.url
    except Exception:
        return url_or_stream


def init_live_state(
    stream_source: str,
    root_dir: Path,
    controls: dict[str, Any],
    normalized_zones: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stream_url = resolve_stream_url(stream_source)
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open livestream source: {stream_source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_area = max(1, frame_width * frame_height)
    zones = []
    if controls["enable_zones"]:
        if normalized_zones:
            zones = build_zones_from_normalized(normalized_zones, frame_width, frame_height)
        else:
            zones = build_default_zones(frame_width, frame_height)

    # Initialize BINTS-inspired bi-modal predictor for livestream
    zone_names = [z["name"] for z in zones] if zones else []
    zone_types = {z["name"]: z["type"] for z in zones} if zones else {}

    return {
        "source": stream_source,
        "stream_url": stream_url,
        "cap": cap,
        "detector": PersonDetector(),
        "track_history": {},
        "tracking_quality": _init_tracking_quality(),
        "flow_state": _init_flow_state(),
        "frame_idx": 0,
        "fps": fps,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_area": frame_area,
        "zones": zones,
        "risk_history": [],
        "bottleneck_risk_history": [],
        "previous_density": 0.0,
        "all_anomalies": [],
        "timeline_rows": [],
        "latest_alerts": [],
        "latest_recommendations": [],
        "latest_zone_metrics": [],
        "bimodal_predictor": BiModalPredictor(zone_names=zone_names, zone_types=zone_types),
    }


def process_live_frame(
    state: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any] | None:
    cap = state["cap"]
    ok, frame = cap.read()
    if not ok:
        return None
    state["frame_idx"] += 1
    state["last_frame_bgr"] = frame.copy()

    tracked_people = state["detector"].track_people(frame, confidence=controls["confidence"])
    sahi_detections = state["detector"].detect_people(frame, confidence=controls["confidence"])
    people_count = max(len(tracked_people), len(sahi_detections))
    _update_track_history(state["track_history"], tracked_people)
    _update_tracking_quality(state["tracking_quality"], tracked_people)
    direction_counts = _direction_distribution(tracked_people)
    entry_exit = _update_flow_counters(tracked_people, state["frame_width"], state["frame_height"], state["flow_state"])
    centroids = [p["centroid"] for p in tracked_people]
    people_count = len(tracked_people)
    density = compute_people_density(people_count, state["frame_area"])
    avg_speed = compute_avg_speed(tracked_people)
    dir_consistency = compute_direction_consistency(tracked_people)
    cluster_pressure = compute_clustering_pressure(tracked_people)

    zone_metrics = compute_zone_metrics(state["zones"], tracked_people, state["frame_area"]) if state["zones"] else []

    # Bi-modal prediction (BINTS-inspired)
    if state["zones"]:
        state["bimodal_predictor"].update(tracked_people, state["zones"])
        zone_predictions = state["bimodal_predictor"].predict()
    else:
        zone_predictions = {}

    max_zone_risk = max((z["local_risk"] for z in zone_metrics), default=0.0)
    max_zone_density = max((z["local_density"] for z in zone_metrics), default=0.0)
    bottleneck_risk = max((z["local_risk"] for z in zone_metrics if z["zone_type"] == "bottleneck"), default=0.0)
    state["bottleneck_risk_history"].append(bottleneck_risk)
    predicted_bottleneck_risk = predict_crowd_stress_risk(state["bottleneck_risk_history"][-20:], horizon_seconds=15)

    anomalies = detect_anomalies(
        frame_idx=state["frame_idx"],
        density=density,
        previous_density=state["previous_density"],
        directional_consistency=dir_consistency,
        avg_speed=avg_speed,
        zone_metrics=zone_metrics,
    )
    state["all_anomalies"].extend(anomalies)

    risk_obj = compute_crowd_stress_risk(
        people_density=density,
        density_growth=density - state["previous_density"],
        zone_congestion=max_zone_risk,
        avg_speed=avg_speed,
        directional_consistency=dir_consistency,
        bottleneck_pressure=cluster_pressure,
        anomaly_count=len(anomalies),
    )
    current_risk = risk_obj["current_risk_score"]
    state["risk_history"].append(current_risk)
    predicted_risk = predict_crowd_stress_risk(state["risk_history"][-15:], horizon_seconds=15)
    state["latest_alerts"] = build_alerts(risk_obj["risk_level"], predicted_risk, anomalies, zone_metrics)
    if predicted_bottleneck_risk >= 0.65:
        state["latest_alerts"].append("Predicted bottleneck escalation in next 15 seconds.")

    # Add bi-modal prediction alerts
    for zname, zpred in zone_predictions.items():
        if zpred.get("time_to_critical") is not None and zpred["time_to_critical"] < 60:
            state["latest_alerts"].append(
                f"BIMODAL: {zname} predicted critical in {zpred['time_to_critical']:.0f}s"
                f" (inflow from {zpred.get('primary_inflow_source', 'unknown')})"
            )
        if zpred.get("cross_modal_alert"):
            state["latest_alerts"].append(f"BIMODAL: Unusual density-flow divergence at {zname}")
        if zpred.get("risk_trend") == "rapidly_increasing":
            state["latest_alerts"].append(f"BIMODAL: {zname} density rapidly increasing")

    state["latest_recommendations"] = build_recommendations(state["latest_alerts"], zone_metrics)
    state["latest_zone_metrics"] = zone_metrics

    annotated = draw_detections(frame, tracked_people) if controls.get("show_boxes", False) else frame.copy()
    if controls["show_heatmap"] and centroids:
        heat = build_heatmap_overlay(frame.shape, centroids)
        annotated = overlay_heatmap(annotated, heat)

    # Compute per-zone direction for live display
    zone_directions = {}
    for zone in state["zones"]:
        zname = zone["name"]
        x1z, y1z, x2z, y2z = zone["rect"]
        zone_people = [p for p in tracked_people if x1z <= p["centroid"][0] <= x2z and y1z <= p["centroid"][1] <= y2z]
        dir_counts = {}
        for p in zone_people:
            traj = p.get("trajectory", [])
            if len(traj) >= 2:
                dx = traj[-1][0] - traj[-2][0]
                dy = traj[-1][1] - traj[-2][1]
                if abs(dx) < 2 and abs(dy) < 2:
                    d = "STILL"
                elif abs(dx) >= abs(dy):
                    d = "EAST" if dx > 0 else "WEST"
                else:
                    d = "SOUTH" if dy > 0 else "NORTH"
                dir_counts[d] = dir_counts.get(d, 0) + 1
        if dir_counts:
            zone_directions[zname] = max(dir_counts, key=dir_counts.get)

    if state["zones"]:
        draw_zones(annotated, state["zones"], zone_metrics=zone_metrics, max_capacity=controls.get("max_capacity", 10), dominant_directions=zone_directions)

    # --- Operator HUD (live) ---
    mc = controls.get("max_capacity", 10)
    danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= mc * 0.8]
    y_cursor = draw_warning_banner(annotated, danger_zones, state["frame_width"], zone_predictions=zone_predictions)
    y_cursor = draw_status_strip(annotated, people_count, zone_metrics, mc, state["frame_width"], y_start=y_cursor)
    if controls.get("show_boxes", False):  # Detail mode
        draw_prediction_detail(annotated, zone_predictions, y_start=y_cursor)

    state["timeline_rows"].append(
        {
            "frame": state["frame_idx"],
            "second": state["frame_idx"] / max(1.0, state["fps"]),
            "people_count": people_count,
            "density": density,
            "avg_speed": avg_speed,
            "directional_consistency": dir_consistency,
            "current_risk": current_risk,
            "predicted_risk": predicted_risk,
            "max_zone_density": max_zone_density,
            "predicted_bottleneck_risk": predicted_bottleneck_risk,
            "entries_total": entry_exit["entries_total"],
            "exits_total": entry_exit["exits_total"],
            "dominant_direction": _dominant_direction(direction_counts),
            "active_tracks": state["tracking_quality"]["active_tracks"],
            "id_switches": state["tracking_quality"]["id_switches"],
            "lost_tracks": state["tracking_quality"]["lost_tracks"],
            "recovered_tracks": state["tracking_quality"]["recovered_tracks"],
            "tracking_stability": _tracking_stability(state["tracking_quality"]),
            "bimodal_predicted_risk": max(
                (zp.get("predicted_density", 0) for zp in zone_predictions.values()),
                default=0.0,
            ) if zone_predictions else 0.0,
        }
    )
    state["previous_density"] = density
    return {
        "frame_bgr": annotated,
        "frame_idx": state["frame_idx"],
        "people_count": people_count,
        "current_risk": current_risk,
        "predicted_risk": predicted_risk,
        "predicted_bottleneck_risk": predicted_bottleneck_risk,
        "zone_predictions": zone_predictions,
    }


def live_state_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    timeline_df = pd.DataFrame(state["timeline_rows"])
    if timeline_df.empty:
        timeline_df = pd.DataFrame(
            [
                {
                    "frame": 0,
                    "second": 0,
                    "people_count": 0,
                    "density": 0,
                    "avg_speed": 0,
                    "current_risk": 0,
                    "predicted_risk": 0,
                    "max_zone_density": 0,
                    "predicted_bottleneck_risk": 0,
                    "entries_total": 0,
                    "exits_total": 0,
                    "dominant_direction": "UNKNOWN",
                    "active_tracks": 0,
                    "id_switches": 0,
                    "lost_tracks": 0,
                    "recovered_tracks": 0,
                    "tracking_stability": 1.0,
                    "bimodal_predicted_risk": 0.0,
                }
            ]
        )
    busiest_zone = "N/A"
    if state["latest_zone_metrics"]:
        busiest_zone = max(state["latest_zone_metrics"], key=lambda z: z["count"])["zone"]
    summary = {
        "session_id": f"live_{uuid.uuid4().hex[:6]}",
        "video_name": "Live Stream",
        "peak_people_count": float(timeline_df["people_count"].max()),
        "latest_current_risk": float(timeline_df["current_risk"].iloc[-1]),
        "latest_predicted_risk": float(timeline_df["predicted_risk"].iloc[-1]),
        "highest_current_risk": float(timeline_df["current_risk"].max()),
        "highest_predicted_risk": float(timeline_df["predicted_risk"].max()),
        "risk_level": _risk_label(float(timeline_df["current_risk"].iloc[-1])),
        "busiest_zone": busiest_zone,
        "predicted_bottleneck_risk": float(timeline_df["predicted_bottleneck_risk"].iloc[-1]),
        "entries_total": int(timeline_df["entries_total"].iloc[-1]),
        "exits_total": int(timeline_df["exits_total"].iloc[-1]),
        "dominant_direction": str(timeline_df["dominant_direction"].iloc[-1]),
        "active_tracks": int(timeline_df["active_tracks"].iloc[-1]),
        "id_switches": int(timeline_df["id_switches"].iloc[-1]),
        "lost_tracks": int(timeline_df["lost_tracks"].iloc[-1]),
        "recovered_tracks": int(timeline_df["recovered_tracks"].iloc[-1]),
        "tracking_stability": float(timeline_df["tracking_stability"].iloc[-1]),
        "zone_predictions": state["bimodal_predictor"].predict() if state.get("bimodal_predictor") else {},
        "flow_matrix": state["bimodal_predictor"].get_flow_matrix() if state.get("bimodal_predictor") else {},
    }
    return {
        "summary": summary,
        "alerts": state["latest_alerts"],
        "recommendations": state["latest_recommendations"],
        "zone_snapshots": state["latest_zone_metrics"],
        "peak_zone_snapshots": state["latest_zone_metrics"],
        "timeline_df": timeline_df,
        "anomalies": state["all_anomalies"],
        "processed_video_path": "",
    }


def _update_track_history(track_history: dict[int, list[tuple[int, int]]], tracked_people: list[dict[str, Any]]) -> None:
    for person in tracked_people:
        tid = int(person.get("track_id", -1))
        if tid < 0:
            person["trajectory"] = [person["centroid"]]
            continue
        history = track_history.setdefault(tid, [])
        history.append(person["centroid"])
        if len(history) > 50:
            del history[:-50]
        person["trajectory"] = history


def _direction_distribution(tracked_people: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for person in tracked_people:
        trajectory = person.get("trajectory", [])
        if len(trajectory) < 2:
            continue
        (x1, y1), (x2, y2) = trajectory[-2], trajectory[-1]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 2 and abs(dy) < 2:
            label = "STILL"
        elif abs(dx) >= abs(dy):
            label = "EAST" if dx > 0 else "WEST"
        else:
            label = "SOUTH" if dy > 0 else "NORTH"
        counts[label] = counts.get(label, 0) + 1
    return counts


def _dominant_direction(direction_counts: dict[str, int]) -> str:
    if not direction_counts:
        return "UNKNOWN"
    return max(direction_counts.items(), key=lambda x: x[1])[0]


def _init_flow_state() -> dict[str, Any]:
    return {
        "entered_ids": set(),
        "exited_ids": set(),
        "entries_total": 0,
        "exits_total": 0,
    }


def _init_tracking_quality() -> dict[str, Any]:
    return {
        "prev_ids": set(),
        "prev_points": {},
        "id_switches": 0,
        "lost_tracks": 0,
        "recovered_tracks": 0,
        "active_tracks": 0,
        "seen_ids": set(),
    }


def _update_tracking_quality(tracking_quality: dict[str, Any], tracked_people: list[dict[str, Any]]) -> None:
    current_points = {int(p["track_id"]): p["centroid"] for p in tracked_people if int(p.get("track_id", -1)) >= 0}
    current_ids = set(current_points.keys())
    prev_ids = tracking_quality["prev_ids"]
    prev_points = tracking_quality["prev_points"]

    # Track missing streaks instead of immediately counting lost
    missing_streaks = tracking_quality.setdefault("missing_streaks", {})

    # Update missing streaks for IDs not in current frame
    for tid in prev_ids - current_ids:
        missing_streaks[tid] = missing_streaks.get(tid, 0) + 1
        if missing_streaks[tid] == 3:  # Only count as lost after 3 consecutive missing frames
            tracking_quality["lost_tracks"] += 1

    # Reset streak for IDs that reappeared
    recovered_now = set()
    for tid in current_ids:
        if tid in missing_streaks and missing_streaks[tid] > 0:
            if missing_streaks[tid] >= 3 and tid in tracking_quality["seen_ids"]:
                tracking_quality["recovered_tracks"] += 1
                recovered_now.add(tid)
            missing_streaks[tid] = 0

    # Clean up old streaks
    for tid in list(missing_streaks.keys()):
        if missing_streaks[tid] > 30:
            del missing_streaks[tid]

    # Approximate ID-switch: nearest previous track was another ID.
    for cid, cpt in current_points.items():
        nearest_prev_id = None
        nearest_dist = float("inf")
        for pid, ppt in prev_points.items():
            dx = cpt[0] - ppt[0]
            dy = cpt[1] - ppt[1]
            d2 = dx * dx + dy * dy
            if d2 < nearest_dist:
                nearest_dist = d2
                nearest_prev_id = pid
        if nearest_prev_id is not None and nearest_prev_id != cid and nearest_dist < 20 * 20:
            tracking_quality["id_switches"] += 1

    tracking_quality["seen_ids"].update(current_ids)
    tracking_quality["prev_ids"] = current_ids
    tracking_quality["prev_points"] = current_points
    tracking_quality["active_tracks"] = len(current_ids)


def _tracking_stability(tracking_quality: dict[str, Any]) -> float:
    active = max(1, tracking_quality["active_tracks"])
    penalty = (tracking_quality["id_switches"] * 0.3 + tracking_quality["lost_tracks"] * 0.05) / (active + 20)
    return max(0.0, min(1.0, 1.0 - penalty))


def _update_flow_counters(
    tracked_people: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
    flow_state: dict[str, Any],
) -> dict[str, int]:
    margin_x = max(10, int(frame_width * 0.05))
    margin_y = max(10, int(frame_height * 0.05))
    for person in tracked_people:
        tid = int(person.get("track_id", -1))
        if tid < 0:
            continue
        trajectory = person.get("trajectory", [])
        if len(trajectory) < 2:
            continue
        (x1, y1), (x2, y2) = trajectory[-2], trajectory[-1]
        dx, dy = x2 - x1, y2 - y1
        near_left = x2 <= margin_x
        near_right = x2 >= frame_width - margin_x
        near_top = y2 <= margin_y
        near_bottom = y2 >= frame_height - margin_y

        # Entering means motion from edge toward interior.
        entering = (near_left and dx > 1) or (near_right and dx < -1) or (near_top and dy > 1) or (near_bottom and dy < -1)
        # Exiting means motion from interior toward edge.
        exiting = (near_left and dx < -1) or (near_right and dx > 1) or (near_top and dy < -1) or (near_bottom and dy > 1)

        if entering and tid not in flow_state["entered_ids"]:
            flow_state["entered_ids"].add(tid)
            flow_state["entries_total"] += 1
        if exiting and tid not in flow_state["exited_ids"]:
            flow_state["exited_ids"].add(tid)
            flow_state["exits_total"] += 1
    return {"entries_total": flow_state["entries_total"], "exits_total": flow_state["exits_total"]}


def _risk_label(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.5:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"