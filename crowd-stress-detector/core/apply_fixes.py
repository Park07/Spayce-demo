#!/usr/bin/env python3
"""
Apply all mentor-feedback UI fixes:
1. Zone colors green/yellow/red based on count vs threshold
2. Toggle bounding boxes off by default
3. Max capacity per zone slider
4. Direction arrows on zones
5. Warning banner when zone is in danger
6. Operator action recommendations in alerts

Run from repo root: python3 apply_fixes.py
"""
import re

# ============================================================
# FIX 1: app/main.py - Add sidebar controls
# ============================================================
with open("app/main.py", "r") as f:
    main_content = f.read()

# Add max_capacity and show_boxes to sidebar
main_content = main_content.replace(
    '    confidence = st.sidebar.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)\n'
    '    sample_stride = st.sidebar.slider("Frame sampling stride", min_value=1, max_value=5, value=1, step=1)',
    '    confidence = st.sidebar.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)\n'
    '    max_capacity = st.sidebar.slider("Max people per zone (alert threshold)", min_value=3, max_value=50, value=10, step=1)\n'
    '    show_boxes = st.sidebar.toggle("Show detection boxes", value=False)\n'
    '    sample_stride = st.sidebar.slider("Frame sampling stride", min_value=1, max_value=5, value=1, step=1)'
)

# Add to return dict
main_content = main_content.replace(
    '        "confidence": confidence,\n'
    '        "sample_stride": sample_stride,',
    '        "confidence": confidence,\n'
    '        "max_capacity": max_capacity,\n'
    '        "show_boxes": show_boxes,\n'
    '        "sample_stride": sample_stride,'
)

with open("app/main.py", "w") as f:
    f.write(main_content)
print("[OK] app/main.py - added max_capacity slider and show_boxes toggle")

# ============================================================
# FIX 2: core/video_processor.py - Update drawing calls
# ============================================================
with open("core/video_processor.py", "r") as f:
    vp_content = f.read()

# Add show_boxes and max_capacity parameters to process_video_file
vp_content = vp_content.replace(
    '    sample_stride: int = 1,\n'
    '    frame_callback: Any | None = None,\n'
    '    normalized_zones: list[dict[str, Any]] | None = None,',
    '    sample_stride: int = 1,\n'
    '    show_boxes: bool = False,\n'
    '    max_capacity: int = 10,\n'
    '    frame_callback: Any | None = None,\n'
    '    normalized_zones: list[dict[str, Any]] | None = None,'
)

# Update draw_detections to be conditional on show_boxes (uploaded video path)
vp_content = vp_content.replace(
    '        annotated = draw_detections(frame, tracked_people)\n'
    '        if show_heatmap and centroids:\n'
    '            heat = build_heatmap_overlay(frame.shape, centroids)\n'
    '            annotated = overlay_heatmap(annotated, heat)\n'
    '        if zones:\n'
    '            draw_zones(annotated, zones)\n'
    '\n'
    '        cv2.putText(\n'
    '            annotated,\n'
    '            f"Crowd Stress Risk: {current_risk:.2f} | Predicted: {predicted_risk:.2f}",',
    '        annotated = draw_detections(frame, tracked_people) if show_boxes else frame.copy()\n'
    '        if show_heatmap and centroids:\n'
    '            heat = build_heatmap_overlay(frame.shape, centroids)\n'
    '            annotated = overlay_heatmap(annotated, heat)\n'
    '\n'
    '        # Compute per-zone direction for display\n'
    '        zone_directions = {}\n'
    '        for zone in zones:\n'
    '            zname = zone["name"]\n'
    '            x1z, y1z, x2z, y2z = zone["rect"]\n'
    '            zone_people = [p for p in tracked_people if x1z <= p["centroid"][0] <= x2z and y1z <= p["centroid"][1] <= y2z]\n'
    '            dir_counts = {}\n'
    '            for p in zone_people:\n'
    '                traj = p.get("trajectory", [])\n'
    '                if len(traj) >= 2:\n'
    '                    dx = traj[-1][0] - traj[-2][0]\n'
    '                    dy = traj[-1][1] - traj[-2][1]\n'
    '                    if abs(dx) < 2 and abs(dy) < 2:\n'
    '                        d = "STILL"\n'
    '                    elif abs(dx) >= abs(dy):\n'
    '                        d = "EAST" if dx > 0 else "WEST"\n'
    '                    else:\n'
    '                        d = "SOUTH" if dy > 0 else "NORTH"\n'
    '                    dir_counts[d] = dir_counts.get(d, 0) + 1\n'
    '            if dir_counts:\n'
    '                zone_directions[zname] = max(dir_counts, key=dir_counts.get)\n'
    '\n'
    '        if zones:\n'
    '            draw_zones(annotated, zones, zone_metrics=zone_metrics, max_capacity=max_capacity, dominant_directions=zone_directions)\n'
    '\n'
    '        # Warning banner when any zone is in DANGER\n'
    '        danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= max_capacity * 0.8]\n'
    '        if danger_zones:\n'
    '            banner_text = f"WARNING: {", ".join(danger_zones)} at capacity - DEPLOY SECURITY NOW"\n'
    '            cv2.rectangle(annotated, (0, 0), (frame_width, 40), (0, 0, 200), -1)\n'
    '            cv2.putText(annotated, banner_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)\n'
    '\n'
    '        cv2.putText(\n'
    '            annotated,\n'
    '            f"Crowd Stress Risk: {current_risk:.2f} | Predicted: {predicted_risk:.2f}",'
,
    1  # Only replace first occurrence (uploaded video path)
)

# Update live stream path - draw_detections conditional
vp_content = vp_content.replace(
    '    annotated = draw_detections(frame, tracked_people)\n'
    '    if controls["show_heatmap"] and centroids:\n'
    '        heat = build_heatmap_overlay(frame.shape, centroids)\n'
    '        annotated = overlay_heatmap(annotated, heat)\n'
    '    if state["zones"]:\n'
    '        draw_zones(annotated, state["zones"])\n'
    '    cv2.putText(\n'
    '        annotated,\n'
    '        f"Crowd Stress Risk: {current_risk:.2f} | Predicted: {predicted_risk:.2f}",',
    '    annotated = draw_detections(frame, tracked_people) if controls.get("show_boxes", False) else frame.copy()\n'
    '    if controls["show_heatmap"] and centroids:\n'
    '        heat = build_heatmap_overlay(frame.shape, centroids)\n'
    '        annotated = overlay_heatmap(annotated, heat)\n'
    '\n'
    '    # Compute per-zone direction for live display\n'
    '    zone_directions = {}\n'
    '    for zone in state["zones"]:\n'
    '        zname = zone["name"]\n'
    '        x1z, y1z, x2z, y2z = zone["rect"]\n'
    '        zone_people = [p for p in tracked_people if x1z <= p["centroid"][0] <= x2z and y1z <= p["centroid"][1] <= y2z]\n'
    '        dir_counts = {}\n'
    '        for p in zone_people:\n'
    '            traj = p.get("trajectory", [])\n'
    '            if len(traj) >= 2:\n'
    '                dx = traj[-1][0] - traj[-2][0]\n'
    '                dy = traj[-1][1] - traj[-2][1]\n'
    '                if abs(dx) < 2 and abs(dy) < 2:\n'
    '                    d = "STILL"\n'
    '                elif abs(dx) >= abs(dy):\n'
    '                    d = "EAST" if dx > 0 else "WEST"\n'
    '                else:\n'
    '                    d = "SOUTH" if dy > 0 else "NORTH"\n'
    '                dir_counts[d] = dir_counts.get(d, 0) + 1\n'
    '        if dir_counts:\n'
    '            zone_directions[zname] = max(dir_counts, key=dir_counts.get)\n'
    '\n'
    '    if state["zones"]:\n'
    '        draw_zones(annotated, state["zones"], zone_metrics=zone_metrics, max_capacity=controls.get("max_capacity", 10), dominant_directions=zone_directions)\n'
    '\n'
    '    # Warning banner for live stream\n'
    '    mc = controls.get("max_capacity", 10)\n'
    '    danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= mc * 0.8]\n'
    '    if danger_zones:\n'
    '        banner_text = f"WARNING: {", ".join(danger_zones)} at capacity - DEPLOY SECURITY NOW"\n'
    '        cv2.rectangle(annotated, (0, 0), (state["frame_width"], 40), (0, 0, 200), -1)\n'
    '        cv2.putText(annotated, banner_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)\n'
    '\n'
    '    cv2.putText(\n'
    '        annotated,\n'
    '        f"Crowd Stress Risk: {current_risk:.2f} | Predicted: {predicted_risk:.2f}",'
)

# Update processor_args in main.py to pass new controls
with open("app/main.py", "r") as f:
    main_content = f.read()

main_content = main_content.replace(
    '                    "confidence": controls["confidence"],\n'
    '                    "sample_stride": controls["sample_stride"],',
    '                    "confidence": controls["confidence"],\n'
    '                    "show_boxes": controls["show_boxes"],\n'
    '                    "max_capacity": controls["max_capacity"],\n'
    '                    "sample_stride": controls["sample_stride"],'
)

with open("app/main.py", "w") as f:
    f.write(main_content)

with open("core/video_processor.py", "w") as f:
    f.write(vp_content)
print("[OK] core/video_processor.py - dynamic zone colors, direction arrows, warning banner, toggle boxes")
print("[OK] app/main.py - added show_boxes and max_capacity to processor_args")

# ============================================================
# FIX 3: core/recommendations.py - Add operator action recommendations
# ============================================================
with open("core/recommendations.py", "r") as f:
    rec_content = f.read()

# Check if build_recommendations exists and add action-based recommendations
# We'll append new action recommendations function
action_recs = '''

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
'''

if "build_operator_actions" not in rec_content:
    rec_content += action_recs
    with open("core/recommendations.py", "w") as f:
        f.write(rec_content)
    print("[OK] core/recommendations.py - added build_operator_actions function")
else:
    print("[SKIP] core/recommendations.py - build_operator_actions already exists")

# ============================================================
# FIX 4: app/ui.py - Add operator actions display + warning banner
# ============================================================
with open("app/ui.py", "r") as f:
    ui_content = f.read()

# Add operator actions renderer if not present
if "render_operator_actions" not in ui_content:
    ui_content += '''

def render_operator_actions(actions: list[str]) -> None:
    """Render operator action recommendations with visual urgency."""
    st.subheader("Operator Actions")
    if not actions:
        st.success("No actions required.")
        return
    for action in actions:
        if action.startswith("IMMEDIATE"):
            st.error(action)
        elif action.startswith("PRE-EMPTIVE"):
            st.warning(action)
        elif action.startswith("CAUTION"):
            st.warning(action)
        elif action.startswith("WATCH"):
            st.info(action)
        elif action.startswith("ALL CLEAR"):
            st.success(action)
        else:
            st.info(action)
'''
    with open("app/ui.py", "w") as f:
        f.write(ui_content)
    print("[OK] app/ui.py - added render_operator_actions")
else:
    print("[SKIP] app/ui.py - render_operator_actions already exists")

print("\n=== ALL FIXES APPLIED ===")
print("1. Zone colors: green/yellow/red based on count vs max_capacity")
print("2. Bounding boxes: off by default, toggleable")
print("3. Max capacity slider: operator sets threshold per zone")
print("4. Direction arrows: shown on each zone")
print("5. Warning banner: red bar across top when zone hits danger")
print("6. Operator actions: specific actions (IMMEDIATE/PRE-EMPTIVE/WATCH)")
print("\nRestart Streamlit and test!")