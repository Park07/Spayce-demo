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


def render_live_overview(session: dict[str, Any], show_trends: bool = True) -> None:
    render_metrics_header(session["summary"])
    col_left, col_right = st.columns([1.2, 1.0], gap="large")
    with col_left:
        render_alerts(session["alerts"])
        render_recommendations(session["recommendations"])
    with col_right:
        render_zone_table(session["zone_snapshots"])

    if show_trends:
        render_trend_charts(session["timeline_df"])


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
    fig_3 = px.line(
        timeline_df,
        x="second",
        y=["max_zone_density", "predicted_bottleneck_risk"],
        title="Zone Congestion and Bottleneck Risk Trend",
        template="plotly_dark",
    )
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(fig_1, use_container_width=True)
    with col2:
        st.plotly_chart(fig_2, use_container_width=True)
    st.plotly_chart(fig_3, use_container_width=True)
