from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def render_metrics_header(summary: dict[str, Any]) -> None:
    st.subheader("Live Session Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Peak People", int(summary["peak_people_count"]))
    c2.metric("Current Crowd Stress Risk", f"{summary['latest_current_risk']:.2f}")
    c3.metric("Predicted Crowd Stress Risk", f"{summary['latest_predicted_risk']:.2f}")
    c4.metric("Highest Current Risk", f"{summary['highest_current_risk']:.2f}")
    c5.metric("Pred. Bottleneck Risk", f"{summary.get('predicted_bottleneck_risk', 0.0):.2f}")
    c6.metric("Entered / Exited", f"{summary.get('entries_total', 0)} / {summary.get('exits_total', 0)}")
    st.markdown(f"<span class='risk-chip'>Risk Level: {summary['risk_level']}</span>", unsafe_allow_html=True)
    st.caption(f"Dominant flow direction: {summary.get('dominant_direction', 'UNKNOWN')}")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Active Tracks", int(summary.get("active_tracks", 0)))
    q2.metric("ID Switches", int(summary.get("id_switches", 0)))
    q3.metric("Lost Tracks", int(summary.get("lost_tracks", 0)))
    q4.metric("Recovered Tracks", int(summary.get("recovered_tracks", 0)))
    q5.metric("Tracking Stability", f"{summary.get('tracking_stability', 1.0):.2f}")


def render_alerts(alerts: list[str]) -> None:
    st.subheader("Alerts")
    if not alerts:
        st.success("No active alerts.")
        return
    for alert in alerts[-8:]:
        st.warning(alert)


def render_recommendations(recommendations: list[str]) -> None:
    st.subheader("Operator Recommendations")
    if not recommendations:
        st.info("No recommendation generated for this segment.")
        return
    for item in recommendations[:8]:
        st.markdown(f"- {item}")


def render_zone_table(zone_snapshots: list[dict[str, Any]]) -> None:
    st.subheader("Zone Breakdown")
    if not zone_snapshots:
        st.info("Zone analysis disabled or no zone activity.")
        return
    df = pd.DataFrame(zone_snapshots)
    if df.empty:
        st.info("No zone observations captured.")
        return
    st.dataframe(df.sort_values(["local_risk", "count"], ascending=False), use_container_width=True, hide_index=True)


def render_zone_predictions(zone_predictions: dict[str, dict[str, Any]]) -> None:
    """Render bi-modal prediction results per zone."""
    st.subheader("Bi-Modal Predictions (BINTS)")
    if not zone_predictions:
        st.info("No prediction data available.")
        return

    for zname, zpred in zone_predictions.items():
        trend = zpred.get("risk_trend", "stable")
        trend_emoji = {
            "stable": "🟢",
            "increasing": "🟡",
            "rapidly_increasing": "🔴",
            "decreasing": "🔵",
            "rapidly_decreasing": "🔵",
        }.get(trend, "⚪")

        ttc = zpred.get("time_to_critical")
        ttc_str = f"{ttc:.0f}s" if ttc is not None else "—"

        with st.container():
            cols = st.columns([1.5, 1, 1, 1, 1])
            cols[0].markdown(f"**{zname}** {trend_emoji} {trend}")
            cols[1].metric("Now", f"{zpred.get('current_density', 0):.0f}")
            cols[2].metric("Predicted", f"{zpred.get('predicted_density', 0):.0f}")
            cols[3].metric("Net Flow", f"{zpred.get('net_flow', 0):+.1f}")
            cols[4].metric("Time to Critical", ttc_str)

        # Show inflow source if present
        src = zpred.get("primary_inflow_source")
        if src and zpred.get("inflow_pressure", 0) > 0:
            st.caption(f"  ↳ Primary inflow from **{src}** (pressure: {zpred['inflow_pressure']:.2f})")

        if zpred.get("cross_modal_alert"):
            st.warning(f"⚠️ Unusual density-flow divergence at {zname}")

    # Bi-modal vs density-only comparison
    st.caption("**Bi-modal advantage** (positive = bi-modal predicts higher risk than density alone):")
    comparison_data = []
    for zname, zpred in zone_predictions.items():
        diff = zpred.get("bimodal_vs_density_only", 0)
        comparison_data.append({"Zone": zname, "Bi-modal vs Density-only": round(diff, 2)})
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)


def render_trend_charts(timeline_df: pd.DataFrame) -> None:
    st.subheader("Trend Graphs")
    if timeline_df.empty:
        st.info("No trend data available.")
        return

    fig_1 = px.line(
        timeline_df,
        x="second",
        y=["people_count", "avg_speed"],
        title="People Count and Movement Speed",
        template="plotly_dark",
    )
    fig_2 = px.line(
        timeline_df,
        x="second",
        y=["current_risk", "predicted_risk"],
        title="Current vs Predicted Crowd Stress Risk",
        template="plotly_dark",
    )

    # Add bi-modal predicted risk if available
    risk_cols = ["max_zone_density", "predicted_bottleneck_risk"]
    if "bimodal_predicted_risk" in timeline_df.columns:
        risk_cols.append("bimodal_predicted_risk")

    fig_3 = px.line(
        timeline_df,
        x="second",
        y=risk_cols,
        title="Zone Congestion, Bottleneck Risk & Bi-Modal Prediction",
        template="plotly_dark",
    )
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(fig_1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_2, use_container_width=True)
    st.plotly_chart(fig_3, use_container_width=True)