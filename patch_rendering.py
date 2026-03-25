#!/usr/bin/env python3
"""Patch v3: Operator-focused UI. Strips engineer-speak from live view."""

# ─── Patch video_processor.py ──────────────────────────────────
with open("core/video_processor.py", "r") as f:
    vp = f.read()

# Fix import line
old_imp = "from core.zones import build_default_zones, build_zones_from_normalized, compute_zone_metrics, draw_zones, draw_warning_banner, draw_status_bar, draw_prediction_panel"
new_imp = "from core.zones import build_default_zones, build_zones_from_normalized, compute_zone_metrics, draw_zones, draw_warning_banner, draw_status_strip, draw_prediction_detail"
if old_imp in vp:
    vp = vp.replace(old_imp, new_imp)
else:
    # Try the original import
    vp = vp.replace(
        "from core.zones import build_default_zones, build_zones_from_normalized, compute_zone_metrics, draw_zones",
        new_imp
    )

# ─── Fix uploaded video drawing ────────────────────────────────
old_hud = """        # --- Clean HUD rendering ---
        danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= max_capacity * 0.8]
        y_cursor = draw_warning_banner(annotated, danger_zones, frame_width)
        y_cursor = draw_status_bar(annotated, current_risk, predicted_risk, people_count, frame_width, y_start=y_cursor)
        draw_prediction_panel(annotated, zone_predictions, y_start=y_cursor)"""

new_hud = """        # --- Operator HUD ---
        danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= max_capacity * 0.8]
        y_cursor = draw_warning_banner(annotated, danger_zones, frame_width, zone_predictions=zone_predictions)
        y_cursor = draw_status_strip(annotated, people_count, zone_metrics, max_capacity, frame_width, y_start=y_cursor)
        if show_boxes:  # Detail mode: show prediction info
            draw_prediction_detail(annotated, zone_predictions, y_start=y_cursor)"""

if old_hud in vp:
    vp = vp.replace(old_hud, new_hud, 1)
    print("[OK] Uploaded video HUD updated")
else:
    print("[WARN] Could not find uploaded video HUD block - trying alternate")
    # Try to find the original cv2.putText block
    if "Crowd Stress Risk:" in vp and "draw_warning_banner" not in vp:
        print("[SKIP] Original drawing code found but no clean HUD - manual fix needed")

# ─── Fix live stream drawing ───────────────────────────────────
old_live = """    # --- Clean HUD rendering (live) ---
    mc = controls.get("max_capacity", 10)
    danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= mc * 0.8]
    y_cursor = draw_warning_banner(annotated, danger_zones, state["frame_width"])
    y_cursor = draw_status_bar(annotated, current_risk, predicted_risk, people_count, state["frame_width"], y_start=y_cursor)
    draw_prediction_panel(annotated, zone_predictions, y_start=y_cursor)"""

new_live = """    # --- Operator HUD (live) ---
    mc = controls.get("max_capacity", 10)
    danger_zones = [zm["zone"] for zm in zone_metrics if zm["count"] >= mc * 0.8]
    y_cursor = draw_warning_banner(annotated, danger_zones, state["frame_width"], zone_predictions=zone_predictions)
    y_cursor = draw_status_strip(annotated, people_count, zone_metrics, mc, state["frame_width"], y_start=y_cursor)
    if controls.get("show_boxes", False):  # Detail mode
        draw_prediction_detail(annotated, zone_predictions, y_start=y_cursor)"""

if old_live in vp:
    vp = vp.replace(old_live, new_live, 1)
    print("[OK] Live stream HUD updated")
else:
    print("[WARN] Could not find live stream HUD block")

with open("core/video_processor.py", "w") as f:
    f.write(vp)

# ─── Patch sidebar in main.py ──────────────────────────────────
with open("app/main.py", "r") as f:
    main = f.read()

# Rename "Show detection boxes" to make it a detail/debug toggle
main = main.replace(
    'show_boxes = st.sidebar.toggle("Show detection boxes", value=False)',
    'show_boxes = st.sidebar.toggle("Show detection boxes + detail", value=False)'
)

# Move debug controls into an expander so they don't eat screen space
old_sidebar = """    st.sidebar.header("Processing Controls")
    show_heatmap = st.sidebar.toggle("Enable heatmap overlay", value=True)
    enable_zones = st.sidebar.toggle("Enable zone analysis", value=True)
    show_trends = st.sidebar.toggle("Show trend charts", value=True)
    enable_multi_camera = st.sidebar.toggle("Mock multi-camera mode", value=False)
    confidence = st.sidebar.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
    max_capacity = st.sidebar.slider("Max people per zone (alert threshold)", min_value=3, max_value=50, value=10, step=1)
    show_boxes = st.sidebar.toggle("Show detection boxes + detail", value=False)
    sample_stride = st.sidebar.slider("Frame sampling stride", min_value=1, max_value=5, value=1, step=1)
    live_speed = st.sidebar.select_slider("Live playback speed", options=["1x", "2x", "4x"], value="1x")"""

new_sidebar = """    st.sidebar.header("Operator Controls")
    max_capacity = st.sidebar.slider("Zone capacity limit", min_value=3, max_value=50, value=10, step=1,
                                      help="Alert triggers when zone reaches 80% of this limit")
    enable_zones = st.sidebar.toggle("Show zones", value=True)
    show_heatmap = st.sidebar.toggle("Heatmap overlay", value=True)
    show_boxes = st.sidebar.toggle("Detail mode (boxes + predictions)", value=False)
    show_trends = st.sidebar.toggle("Show trend charts", value=True)

    with st.sidebar.expander("Advanced Settings"):
        confidence = st.sidebar.slider("Detection confidence", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
        sample_stride = st.sidebar.slider("Frame sampling stride", min_value=1, max_value=5, value=1, step=1)
        enable_multi_camera = st.sidebar.toggle("Mock multi-camera mode", value=False)
        live_speed = st.sidebar.select_slider("Live playback speed", options=["1x", "2x", "4x"], value="1x")"""

if old_sidebar in main:
    main = main.replace(old_sidebar, new_sidebar)
    print("[OK] Sidebar simplified to operator-focused")
else:
    print("[WARN] Could not find sidebar block - check main.py manually")

with open("app/main.py", "w") as f:
    f.write(main)

print("\n=== V3 PATCHES APPLIED ===")
print("Changes:")
print("  - Banner: shows operator ACTION (restrict entry at X, deploy security)")
print("  - Status strip: thin bar with people count + colored zone dots")
print("  - Engineer metrics hidden (risk/predicted/ttc) unless Detail mode ON")
print("  - Sidebar: 'Operator Controls' with capacity limit as #1 control")
print("  - Debug settings in 'Advanced' expander")
print("  - Zone labels: operator language (SAFE/BUSY/CROWDED/CRITICAL)")
print("  - Prediction detail only shows in Detail mode (Show detection boxes)")
print("\nRestart Streamlit and test!")