from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.theme import apply_dark_theme
from app.ui import (
    render_alerts,
    render_metrics_header,
    render_operator_actions,
    render_recommendations,
    render_trend_charts,
    render_zone_predictions,
    render_zone_table,
)
from core.config import APP_TITLE, VIDEO_EXTENSIONS
from core.export import build_session_pdf
from core.recommendations import build_recommendations
from core.utils import save_uploaded_file
from core.video_processor import init_live_state, live_state_snapshot, process_live_frame, process_video_file
from core.zones import build_zones_from_normalized, default_zone_templates

try:
    from core.recommendations import build_operator_actions
except ImportError:
    build_operator_actions = None


def _sample_video_options() -> list[Path]:
    sample_dir = ROOT_DIR / "data" / "sample_videos"
    if not sample_dir.exists():
        return []
    paths: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(sample_dir.glob(f"*.{ext}"))
    return sorted(paths)


def _sidebar_controls() -> dict[str, Any]:
    st.sidebar.markdown("### Operator Controls")
    max_capacity = st.sidebar.slider("Zone capacity limit", min_value=3, max_value=50, value=10, step=1,
                                      help="Alert when zone reaches 80% of this limit")
    whole_frame_zone = st.sidebar.toggle("Whole frame as single zone", value=True,
                                          help="Monitor entire camera as one zone. Turn off for custom zones.")
    enable_zones = st.sidebar.toggle("Show zones", value=True)
    show_heatmap = st.sidebar.toggle("Heatmap overlay", value=True)
    show_boxes = st.sidebar.toggle("Detail mode", value=False,
                                    help="Show detection boxes with IDs and prediction details")
    show_trends = st.sidebar.toggle("Trend charts", value=True)

    with st.sidebar.expander("Advanced", expanded=False):
        confidence = st.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        sample_stride = st.slider("Frame stride", min_value=1, max_value=5, value=1, step=1)
        enable_multi_camera = st.toggle("Multi-camera mock", value=False)
        live_speed = st.select_slider("Playback speed", options=["1x", "2x", "4x"], value="1x")

    return {
        "show_heatmap": show_heatmap,
        "enable_zones": enable_zones,
        "show_trends": show_trends,
        "enable_multi_camera": enable_multi_camera,
        "confidence": confidence,
        "max_capacity": max_capacity,
        "show_boxes": show_boxes,
        "sample_stride": sample_stride,
        "live_speed": live_speed,
        "whole_frame_zone": whole_frame_zone,
    }


def _read_preview_frame(video_path: Path, frame_ratio: float) -> tuple[Any, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_idx = int(max(0, total_frames - 1) * frame_ratio) if total_frames > 0 else 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        return None, total_frames, target_idx
    return frame, total_frames, target_idx


def _zone_editor(video_key: str, input_path: Path) -> list[dict[str, Any]]:
    state_key = f"zone_editor_{video_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [dict(z) for z in default_zone_templates()]

    st.subheader("Zone Editor (Pause and Configure)")
    frame_ratio = st.slider("Paused frame position", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    frame, total_frames, target_idx = _read_preview_frame(input_path, frame_ratio)
    if frame is None:
        st.warning("Unable to load preview frame for zone editing.")
        return st.session_state[state_key]

    zones = st.session_state[state_key]
    for idx, zone in enumerate(zones):
        with st.expander(f"{zone['type'].upper()} - {zone['name']}", expanded=(idx == 0)):
            new_name = st.text_input("Zone label", value=zone["name"], key=f"{video_key}_{idx}_name")
            zone_type = st.selectbox(
                "Zone type",
                options=["exit", "entry", "bottleneck"],
                index=["exit", "entry", "bottleneck"].index(zone["type"]),
                key=f"{video_key}_{idx}_type",
            )
            x1, y1, x2, y2 = zone["rect"]
            x_range = st.slider("X range (normalized)", 0.0, 1.0, (float(x1), float(x2)), 0.01, key=f"{video_key}_{idx}_x")
            y_range = st.slider("Y range (normalized)", 0.0, 1.0, (float(y1), float(y2)), 0.01, key=f"{video_key}_{idx}_y")
            zone["name"] = new_name.strip() or zone["name"]
            zone["type"] = zone_type
            zone["rect"] = (min(x_range), min(y_range), max(x_range), max(y_range))
            zones[idx] = zone

    st.session_state[state_key] = zones
    h, w = frame.shape[:2]
    preview = frame.copy()
    for zone in zones:
        x1n, y1n, x2n, y2n = zone["rect"]
        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
        color = (0, 220, 255) if zone["type"] == "exit" else (0, 255, 120) if zone["type"] == "entry" else (160, 160, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, zone["name"], (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    st.caption(f"Paused on frame {target_idx}/{max(total_frames, 1)} for zone editing.")
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    return zones


def _zone_editor_from_frame(video_key: str, frame_bgr, initial_zones: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    state_key = f"zone_editor_{video_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = [dict(z) for z in (initial_zones or default_zone_templates())]

    st.subheader("Zone Editor (Paused Live Stream)")
    zones = st.session_state[state_key]
    for idx, zone in enumerate(zones):
        with st.expander(f"{zone['type'].upper()} - {zone['name']}", expanded=(idx == 0)):
            new_name = st.text_input("Zone label", value=zone["name"], key=f"{video_key}_{idx}_name")
            zone_type = st.selectbox(
                "Zone type",
                options=["exit", "entry", "bottleneck"],
                index=["exit", "entry", "bottleneck"].index(zone["type"]),
                key=f"{video_key}_{idx}_type",
            )
            x1, y1, x2, y2 = zone["rect"]
            x_range = st.slider("X range (normalized)", 0.0, 1.0, (float(x1), float(x2)), 0.01, key=f"{video_key}_{idx}_x")
            y_range = st.slider("Y range (normalized)", 0.0, 1.0, (float(y1), float(y2)), 0.01, key=f"{video_key}_{idx}_y")
            zones[idx] = {
                "name": new_name.strip() or zone["name"],
                "type": zone_type,
                "rect": (min(x_range), min(y_range), max(x_range), max(y_range)),
            }
    st.session_state[state_key] = zones

    h, w = frame_bgr.shape[:2]
    preview = frame_bgr.copy()
    for zone in zones:
        x1n, y1n, x2n, y2n = zone["rect"]
        x1, y1, x2, y2 = int(x1n * w), int(y1n * h), int(x2n * w), int(y2n * h)
        color = (0, 220, 255) if zone["type"] == "exit" else (0, 255, 120) if zone["type"] == "entry" else (160, 160, 255)
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview, zone["name"], (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    return zones


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    apply_dark_theme()

    st.markdown(f"## {APP_TITLE}")
    st.caption("Predictive crowd intelligence for safer venues.")

    controls = _sidebar_controls()
    live_speed = controls["live_speed"]
    live_render_stride = {"1x": 1, "2x": 2, "4x": 4}.get(live_speed, 1)
    source_mode = st.radio("Source", options=["Uploaded Video", "Live Stream"], horizontal=True)

    input_path = None
    source_key = "unknown_source"
    normalized_zones = None
    if controls.get("whole_frame_zone", True):
        normalized_zones = [{"name": "Camera View", "type": "bottleneck", "rect": (0.0, 0.0, 1.0, 1.0)}]
    session = st.session_state.get("last_session")

    if source_mode == "Uploaded Video":
        uploaded_video = st.file_uploader("Upload crowd video", type=VIDEO_EXTENSIONS)
        sample_videos = _sample_video_options()
        selected_sample_name = st.selectbox(
            "Or choose a bundled sample video",
            options=["None"] + [p.name for p in sample_videos],
            index=0,
        )
        selected_sample_path = None
        if selected_sample_name != "None":
            selected_sample_path = next((p for p in sample_videos if p.name == selected_sample_name), None)

        if uploaded_video is None and selected_sample_path is None:
            st.info("Upload a crowd video or select a sample to begin analysis.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Risk", "0.00")
            c2.metric("Predicted Risk", "0.00")
            c3.metric("Peak People", "0")
            return

        if uploaded_video is not None:
            upload_signature = f"{uploaded_video.name}:{uploaded_video.size}"
            cached_signature = st.session_state.get("uploaded_video_signature")
            cached_path = st.session_state.get("uploaded_video_path")
            if cached_signature != upload_signature or not cached_path:
                saved_path = save_uploaded_file(uploaded_video, ROOT_DIR / "data" / "sample_videos")
                st.session_state["uploaded_video_signature"] = upload_signature
                st.session_state["uploaded_video_path"] = str(saved_path)
                input_path = saved_path
            else:
                input_path = Path(cached_path)
        else:
            input_path = selected_sample_path
            st.session_state.pop("uploaded_video_signature", None)
            st.session_state.pop("uploaded_video_path", None)
        source_key = input_path.name if input_path is not None else "uploaded_source"

        if controls["enable_zones"] and input_path is not None:
            pass

        process_btn = st.button("Process Session", type="primary", use_container_width=True)
        if process_btn:
            st.session_state["last_session"] = None
            with st.spinner("Analyzing crowd dynamics..."):
                processor_args = {
                    "input_path": input_path,
                    "root_dir": ROOT_DIR,
                    "show_heatmap": controls["show_heatmap"],
                    "enable_zones": controls["enable_zones"],
                    "show_trends": controls["show_trends"],
                    "enable_multi_camera": controls["enable_multi_camera"],
                    "confidence": controls["confidence"],
                    "show_boxes": controls["show_boxes"],
                    "max_capacity": controls["max_capacity"],
                    "sample_stride": controls["sample_stride"],
                    "normalized_zones": normalized_zones,
                }
                live_frame = st.empty()
                progress_text = st.empty()
                progress_bar = st.progress(0)

                def _frame_callback(
                    frame_bgr,
                    frame_idx: int,
                    total_frames: int,
                    current_risk: float,
                    predicted_risk: float,
                    people_count: int,
                ) -> None:
                    if frame_idx % live_render_stride != 0:
                        return
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    live_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                    if total_frames > 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    progress_text.caption(
                        f"Frame {frame_idx}/{max(total_frames, frame_idx)} | "
                        f"People: {people_count} | "
                        f"Risk: {current_risk:.2f}"
                    )

                try:
                    session = process_video_file(**processor_args, frame_callback=_frame_callback)
                except TypeError as exc:
                    if "frame_callback" in str(exc):
                        session = process_video_file(**processor_args)
                    else:
                        raise
                progress_bar.progress(1.0)
                progress_text.caption("Processing complete.")
            st.session_state["last_session"] = session

    else:
        stream_source = st.text_input(
            "Live stream URL or page URL",
            value="https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/fontana-di-trevi.html",
        )
        live_key = f"live:{stream_source.strip()}"
        st.session_state.setdefault("live_running", False)
        st.session_state.setdefault("live_paused", True)

        c1, c2, c3 = st.columns(3)
        if c1.button("Start Live Analysis", type="primary"):
            live_zones = st.session_state.get(f"zone_editor_{live_key}")
            st.session_state["live_state"] = init_live_state(stream_source, ROOT_DIR, controls, live_zones)
            st.session_state["live_running"] = True
            st.session_state["live_paused"] = False
        if c2.button("Pause"):
            st.session_state["live_paused"] = True
        if c3.button("Resume"):
            st.session_state["live_paused"] = False

        if st.session_state.get("live_running") and st.session_state.get("live_state") is not None:
            state = st.session_state["live_state"]
            frame_holder = st.empty()
            if not st.session_state["live_paused"]:
                payload = process_live_frame(state, controls)
                if payload is not None:
                    frame_rgb = cv2.cvtColor(payload["frame_bgr"], cv2.COLOR_BGR2RGB)
                    frame_holder.image(frame_rgb, channels="RGB", use_container_width=True)
                    st.caption(
                        f"Live Frame {payload['frame_idx']} | People: {payload['people_count']} | "
                        f"Risk: {payload['current_risk']:.2f}"
                    )
                st.session_state["last_session"] = live_state_snapshot(state)
                time.sleep(0.04 if live_speed == "1x" else 0.02 if live_speed == "2x" else 0.01)
                st.rerun()
            else:
                st.info("Live analysis paused.")
                last_frame = state.get("last_frame_bgr")
                if controls["enable_zones"] and last_frame is not None:
                    edited = _zone_editor_from_frame(live_key, last_frame)
                    st.session_state[f"zone_editor_{live_key}"] = edited
                    state["zones"] = build_zones_from_normalized(
                        edited,
                        int(last_frame.shape[1]),
                        int(last_frame.shape[0]),
                    )
        session = st.session_state.get("last_session")

    # ─── Results Dashboard ──────────────────────────────────────
    if session is None:
        return

    st.markdown("---")

    # ── Critical alert popup with shake ──
    zone_preds_check = session.get("summary", {}).get("zone_predictions", {})
    critical_zones = []
    for zn, zp in zone_preds_check.items():
        ttc = zp.get("time_to_critical")
        if zp.get("risk_trend") == "rapidly_increasing" or (ttc is not None and ttc < 120):
            critical_zones.append((zn, zp))

    if critical_zones:
        zn, zp = critical_zones[0]
        src = zp.get("primary_inflow_source", "adjacent zone")
        ttc = zp.get("time_to_critical")
        ttc_str = f"in {ttc:.0f}s" if ttc else "imminently"

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
            border: 2px solid #ef4444;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 16px;
            animation: pulse-border 2s ease-in-out infinite;
        ">
            <div style="font-family: 'Inter',sans-serif; font-weight: 800; font-size: 1.1rem; color: #ef4444; margin-bottom: 6px;">
                CRITICAL — {zn} approaching capacity {ttc_str}
            </div>
            <div style="font-family: 'Inter',sans-serif; font-size: 0.9rem; color: #e8ecf4; margin-bottom: 10px;">
                Restrict inflow at <strong>{src}</strong> — Deploy security to <strong>{zn}</strong> — Announce redistribution on PA
            </div>
            <div style="font-family: 'JetBrains Mono',monospace; font-size: 0.75rem; color: #5a6a80;">
                Predicted: {zp.get('predicted_density', 0):.0f} people | Trend: {zp.get('risk_trend', 'unknown')} | Net flow: {zp.get('net_flow', 0):+.1f}
            </div>
        </div>
        <style>
            @keyframes pulse-border {{
                0%, 100% {{ border-color: #ef4444; }}
                50% {{ border-color: #991b1b; }}
            }}
        </style>
        """, unsafe_allow_html=True)

        if st.button("ACKNOWLEDGE ALERT — DISPATCH SECURITY", key="ack_alert", type="primary", use_container_width=True):
            st.markdown("""
            <style>
                @keyframes violent-shake {
                    0% { transform: translate(0) rotate(0deg); }
                    2% { transform: translate(-12px, 8px) rotate(-2deg); }
                    4% { transform: translate(12px, -8px) rotate(2deg); }
                    6% { transform: translate(-10px, -12px) rotate(-2.5deg); }
                    8% { transform: translate(10px, 12px) rotate(2.5deg); }
                    10% { transform: translate(-12px, -6px) rotate(-1.5deg); }
                    12% { transform: translate(12px, 6px) rotate(1.5deg); }
                    14% { transform: translate(-8px, 12px) rotate(-3deg); }
                    16% { transform: translate(8px, -12px) rotate(3deg); }
                    18% { transform: translate(-12px, 10px) rotate(-2deg); }
                    20% { transform: translate(12px, -10px) rotate(2deg); }
                    22% { transform: translate(-10px, -8px) rotate(-2.5deg); }
                    24% { transform: translate(10px, 8px) rotate(2.5deg); }
                    26% { transform: translate(-8px, 12px) rotate(-1.5deg); }
                    28% { transform: translate(8px, -12px) rotate(1.5deg); }
                    30% { transform: translate(-12px, 6px) rotate(-3deg); }
                    32% { transform: translate(12px, -6px) rotate(3deg); }
                    34% { transform: translate(-10px, 10px) rotate(-2deg); }
                    36% { transform: translate(10px, -10px) rotate(2deg); }
                    38% { transform: translate(-8px, -12px) rotate(-2.5deg); }
                    40% { transform: translate(8px, 12px) rotate(2.5deg); }
                    42% { transform: translate(-12px, -8px) rotate(-1.5deg); }
                    44% { transform: translate(12px, 8px) rotate(1.5deg); }
                    46% { transform: translate(-10px, 6px) rotate(-3deg); }
                    48% { transform: translate(10px, -6px) rotate(3deg); }
                    50% { transform: translate(-8px, 10px) rotate(-2deg); }
                    52% { transform: translate(8px, -10px) rotate(2deg); }
                    54% { transform: translate(-12px, -6px) rotate(-2.5deg); }
                    56% { transform: translate(12px, 6px) rotate(2.5deg); }
                    58% { transform: translate(-10px, 12px) rotate(-1.5deg); }
                    60% { transform: translate(10px, -12px) rotate(1.5deg); }
                    62% { transform: translate(-8px, 8px) rotate(-3deg); }
                    64% { transform: translate(8px, -8px) rotate(3deg); }
                    66% { transform: translate(-12px, 10px) rotate(-2deg); }
                    68% { transform: translate(12px, -10px) rotate(2deg); }
                    70% { transform: translate(-10px, -12px) rotate(-2.5deg); }
                    72% { transform: translate(10px, 12px) rotate(2.5deg); }
                    74% { transform: translate(-8px, -6px) rotate(-1.5deg); }
                    76% { transform: translate(8px, 6px) rotate(1.5deg); }
                    78% { transform: translate(-12px, 12px) rotate(-2deg); }
                    80% { transform: translate(10px, -8px) rotate(1.5deg); }
                    82% { transform: translate(-8px, 6px) rotate(-1deg); }
                    84% { transform: translate(6px, -6px) rotate(1deg); }
                    86% { transform: translate(-6px, 4px) rotate(-0.5deg); }
                    88% { transform: translate(4px, -4px) rotate(0.5deg); }
                    90% { transform: translate(-4px, 3px) rotate(-0.3deg); }
                    92% { transform: translate(3px, -3px) rotate(0.3deg); }
                    94% { transform: translate(-2px, 2px) rotate(-0.2deg); }
                    96% { transform: translate(2px, -2px) rotate(0.2deg); }
                    98% { transform: translate(-1px, 1px) rotate(-0.1deg); }
                    100% { transform: translate(0) rotate(0deg); }
                }
                @keyframes red-strobe {
                    0%, 100% { background: rgba(239, 68, 68, 0.0); }
                    50% { background: rgba(239, 68, 68, 0.15); }
                }
                [data-testid="stAppViewContainer"] {
                    animation: violent-shake 5s cubic-bezier(0.36, 0.07, 0.19, 0.97) !important;
                }
                [data-testid="stAppViewContainer"]::before {
                    content: '';
                    position: fixed;
                    top: 0; left: 0; right: 0; bottom: 0;
                    background: rgba(239, 68, 68, 0.0);
                    animation: red-strobe 0.15s ease-in-out 20;
                    pointer-events: none;
                    z-index: 99999;
                }
            </style>
            """, unsafe_allow_html=True)
            st.toast("Alert acknowledged. Security team dispatched.", icon="🚨")

    # ── Row 1: Metrics ──
    render_metrics_header(session["summary"])

    # ── Row 2: Video + Actions ──
    vid_col, action_col = st.columns([2.2, 1], gap="large")

    with vid_col:
        processed_path = session.get("processed_video_path")
        if processed_path:
            st.subheader("Processed Feed")
            st.video(str(processed_path))

    with action_col:
        zone_preds = session.get("summary", {}).get("zone_predictions", {})
        zone_data = session.get("peak_zone_snapshots", session.get("zone_snapshots", []))

        if build_operator_actions is not None and zone_data:
            actions = build_operator_actions(zone_data, zone_preds, controls.get("max_capacity", 10))
            render_operator_actions(actions)
        else:
            render_alerts(session.get("alerts", []))

        render_recommendations(session.get("recommendations", []))

    # ── Row 3: Zones + Predictions ──
    zone_col, pred_col = st.columns([1, 1.2], gap="large")

    with zone_col:
        zone_display = session.get("peak_zone_snapshots", session.get("zone_snapshots", []))
        render_zone_table(zone_display)

    with pred_col:
        zone_preds = session.get("summary", {}).get("zone_predictions", {})
        if zone_preds:
            render_zone_predictions(zone_preds)

    # ── Row 4: Alert log ──
    if build_operator_actions is not None:
        alerts = session.get("alerts", [])
        if alerts:
            with st.expander(f"Alert Log ({len(alerts)} alerts)", expanded=False):
                render_alerts(alerts)

    # ── Row 5: Trends ──
    if controls["show_trends"]:
        render_trend_charts(session.get("timeline_df"))

    # ── Export ──
    st.markdown("---")
    report_bytes = build_session_pdf(session)
    st.download_button(
        "Download PDF Report",
        data=report_bytes,
        file_name=f"{session['summary']['session_id']}_report.pdf",
        mime="application/pdf",
    )


if __name__ == "__main__":
    main()