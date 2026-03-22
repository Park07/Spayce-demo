from __future__ import annotations

import sys
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
    render_live_overview,
    render_metrics_header,
    render_recommendations,
    render_trend_charts,
    render_zone_table,
)
from core.config import APP_TITLE, VIDEO_EXTENSIONS
from core.export import build_session_pdf
from core.utils import save_uploaded_file
from core.video_processor import init_live_state, live_state_snapshot, process_live_frame, process_video_file
from core.zones import build_zones_from_normalized, default_zone_templates


def _sample_video_options() -> list[Path]:
    sample_dir = ROOT_DIR / "data" / "sample_videos"
    if not sample_dir.exists():
        return []
    paths: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        paths.extend(sample_dir.glob(f"*.{ext}"))
    return sorted(paths)


def _sidebar_controls() -> dict[str, Any]:
    st.sidebar.header("Processing Controls")
    show_heatmap = st.sidebar.toggle("Enable heatmap overlay", value=True)
    enable_zones = st.sidebar.toggle("Enable zone analysis", value=True)
    show_trends = st.sidebar.toggle("Show trend charts", value=True)
    enable_multi_camera = st.sidebar.toggle("Mock multi-camera mode", value=False)
    confidence = st.sidebar.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    sample_stride = st.sidebar.slider("Frame sampling stride", min_value=1, max_value=5, value=1, step=1)
    live_speed = st.sidebar.select_slider("Live playback speed", options=["1x", "2x", "4x"], value="1x")
    auto_start_live = st.sidebar.toggle("Auto-start live overview", value=True)
    return {
        "show_heatmap": show_heatmap,
        "enable_zones": enable_zones,
        "show_trends": show_trends,
        "enable_multi_camera": enable_multi_camera,
        "confidence": confidence,
        "sample_stride": sample_stride,
        "live_speed": live_speed,
        "auto_start_live": auto_start_live,
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


def _live_refresh_interval(live_speed: str) -> str:
    return {"1x": "2s", "2x": "1s", "4x": "500ms"}.get(live_speed, "1s")


def _ensure_live_session(
    stream_source: str,
    controls: dict[str, Any],
    source_key: str,
    normalized_zones: list[dict[str, Any]] | None = None,
) -> bool:
    existing_key = st.session_state.get("live_source_key")
    live_state = st.session_state.get("live_state")
    if live_state is not None and existing_key == source_key:
        st.session_state["live_running"] = True
        return True

    if live_state is not None and existing_key != source_key:
        try:
            live_state["cap"].release()
        except Exception:
            pass

    try:
        live_zones = normalized_zones or st.session_state.get(f"zone_editor_{source_key}")
        st.session_state["live_state"] = init_live_state(stream_source, ROOT_DIR, controls, live_zones)
        st.session_state["live_source_key"] = source_key
        st.session_state["live_running"] = True
        st.session_state["live_paused"] = False
        st.session_state["last_session"] = live_state_snapshot(st.session_state["live_state"])
        return True
    except Exception as exc:
        st.error(f"Unable to open live stream: {exc}")
        st.session_state["live_running"] = False
        return False


def _render_live_overview_fragment(controls: dict[str, Any], source_key: str) -> None:
    refresh_every = _live_refresh_interval(controls["live_speed"])

    @st.fragment(run_every=refresh_every)
    def _render() -> None:
        live_state = st.session_state.get("live_state")
        if not st.session_state.get("live_running") or live_state is None:
            st.info("Live overview is ready. Start analysis to begin auto-refreshing the screen.")
            return

        st.caption("Auto-refreshing live overview. Pause to inspect the latest frame or adjust zones.")
        action_left, action_right, action_reinit = st.columns([1, 1, 2])
        if action_left.button("Pause", key=f"{source_key}_pause"):
            st.session_state["live_paused"] = True
        if action_right.button("Resume", key=f"{source_key}_resume"):
            st.session_state["live_paused"] = False
        if action_reinit.button("Reinitialize live session", key=f"{source_key}_reinit"):
            st.session_state.pop("live_state", None)
            st.session_state["live_running"] = False
            _ensure_live_session(
                st.session_state.get("live_stream_source", ""),
                controls,
                source_key,
                st.session_state.get(f"zone_editor_{source_key}"),
            )

        state = st.session_state.get("live_state")
        if state is None:
            st.warning("Live session is not available yet.")
            return

        if st.session_state.get("live_paused"):
            st.info("Live analysis paused.")
            last_frame = state.get("last_frame_bgr")
            if controls["enable_zones"] and last_frame is not None:
                edited = _zone_editor_from_frame(source_key, last_frame, state.get("zones"))
                st.session_state[f"zone_editor_{source_key}"] = edited
                state["zones"] = build_zones_from_normalized(
                    edited,
                    int(last_frame.shape[1]),
                    int(last_frame.shape[0]),
                )
            snapshot = live_state_snapshot(state)
            render_live_overview(snapshot, show_trends=controls["show_trends"])
            return

        payload = process_live_frame(state, controls)
        if payload is None:
            st.warning("Live feed yielded no frame yet. Waiting for the next update.")
            return

        frame_rgb = cv2.cvtColor(payload["frame_bgr"], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_container_width=True)
        st.caption(
            f"Live Frame {payload['frame_idx']} | People: {payload['people_count']} | "
            f"Current Crowd Stress Risk: {payload['current_risk']:.2f} | "
            f"Predicted: {payload['predicted_risk']:.2f} | "
            f"Pred. Bottleneck: {payload.get('predicted_bottleneck_risk', 0.0):.2f}"
        )
        session = live_state_snapshot(state)
        st.session_state["last_session"] = session
        render_live_overview(session, show_trends=controls["show_trends"])

    _render()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    apply_dark_theme()
    st.title(APP_TITLE)
    st.caption("Predictive crowd intelligence for safer venues.")

    controls = _sidebar_controls()
    live_speed = controls["live_speed"]
    live_render_stride = {"1x": 1, "2x": 2, "4x": 4}.get(live_speed, 1)
    source_mode = st.radio("Source", options=["Uploaded Video", "Live Stream"], horizontal=True)

    input_path = None
    source_key = "unknown_source"
    normalized_zones = None
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
            st.info("No video loaded yet. Upload a crowd video (or pick a sample).")
            st.subheader("Ready State")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Crowd Stress Risk", "0.00")
            c2.metric("Predicted Crowd Stress Risk", "0.00")
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
            normalized_zones = _zone_editor(source_key, input_path)

        process_btn = st.button("Process Session", type="primary")
        if process_btn:
            st.session_state["last_session"] = None
            with st.spinner("Analyzing crowd dynamics in live mode..."):
                processor_args = {
                    "input_path": input_path,
                    "root_dir": ROOT_DIR,
                    "show_heatmap": controls["show_heatmap"],
                    "enable_zones": controls["enable_zones"],
                    "show_trends": controls["show_trends"],
                    "enable_multi_camera": controls["enable_multi_camera"],
                    "confidence": controls["confidence"],
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
                        f"Current Crowd Stress Risk: {current_risk:.2f} | "
                        f"Predicted: {predicted_risk:.2f}"
                    )

                try:
                    session = process_video_file(**processor_args, frame_callback=_frame_callback)
                except TypeError as exc:
                    if "frame_callback" in str(exc):
                        session = process_video_file(**processor_args)
                        st.warning("Live feed callback was unavailable in the current run. Please refresh the page once.")
                    else:
                        raise
                progress_bar.progress(1.0)
                progress_text.caption("Processing complete.")
            st.session_state["last_session"] = session
        else:
            st.caption("Adjust zones, then click Process Session.")

    else:
        stream_source = st.text_input(
            "Live stream URL or page URL",
            value="https://www.skylinewebcams.com/en/webcam/italia/lazio/roma/fontana-di-trevi.html",
        )
        st.session_state["live_stream_source"] = stream_source
        live_key = f"live:{stream_source.strip()}"
        start_requested = controls["auto_start_live"]
        start_col, reset_col = st.columns(2)
        if start_col.button("Start Live Overview", type="primary"):
            start_requested = True
        if reset_col.button("Reset Live Session"):
            current_state = st.session_state.get("live_state")
            if current_state is not None:
                try:
                    current_state["cap"].release()
                except Exception:
                    pass
            st.session_state.pop("live_state", None)
            st.session_state.pop("last_session", None)
            st.session_state["live_running"] = False
            st.session_state["live_paused"] = False

        live_active = (
            st.session_state.get("live_running")
            and st.session_state.get("live_state") is not None
            and st.session_state.get("live_source_key") == live_key
        )
        if start_requested and not live_active:
            live_active = _ensure_live_session(stream_source, controls, live_key)

        if live_active:
            _render_live_overview_fragment(controls, live_key)
            return

        st.info("Set the live source above and start the overview to begin auto-refreshing the screen.")
        session = None

    if session is None:
        return

    render_metrics_header(session["summary"])

    processed_path = session.get("processed_video_path")
    if processed_path:
        st.subheader("Processed Video")
        st.video(str(processed_path))

    col_left, col_right = st.columns([1.2, 1.0], gap="large")
    with col_left:
        render_alerts(session["alerts"])
        render_recommendations(session["recommendations"])
    with col_right:
        render_zone_table(session["zone_snapshots"])

    if controls["show_trends"]:
        render_trend_charts(session["timeline_df"])

    report_bytes = build_session_pdf(session)
    st.download_button(
        "Download PDF Session Summary",
        data=report_bytes,
        file_name=f"{session['summary']['session_id']}_summary.pdf",
        mime="application/pdf",
    )

<<<<<<< Updated upstream
=======
    st.markdown("---")
    st.header("Agentic Control Center")
    if st.button("Generate Live LLM Analysis", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing privacy-blurred footage and telemetry..."):
             # In a real app we pass the actual anonymized frame here,
             # for Streamlit MVP we pass a blank frame or the last processed path frame
             import numpy as np
             dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
             
             # Create the JSON telemetry payload from the session summary
             telemetry = session["summary"]
             telemetry["active_anomalies"] = session.get("anomalies", [])[-3:] # last 3 anomalies
             
             try:
                 # Try to use live Gemini API
                 llm = CrowdSafetyLLM()
                 ai_payload = llm.analyze_scene(dummy_frame, telemetry)
             except ValueError as e:
                 # API key not set - use mock mode
                 st.warning("⚠️ Gemini API key not configured. Using mock analysis mode.")
                 llm = CrowdSafetyLLM(use_mock=True)
                 ai_payload = llm.analyze_scene(dummy_frame, telemetry)
             except Exception as e:
                 st.error(f"❌ LLM Error: {str(e)}")
                 ai_payload = {
                     "reasoning": "Error initializing LLM",
                     "description": str(e),
                     "risk_level": "Unknown",
                     "action": "Set GEMINI_API_KEY environment variable to enable live analysis"
                 }
             
             render_ai_assistant(ai_payload)

>>>>>>> Stashed changes

if __name__ == "__main__":
    main()
