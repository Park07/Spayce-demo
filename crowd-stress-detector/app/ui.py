"""
Crowd Stress Detector — Operations Center UI
Professional dashboard rendering with semantic color system.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ─── Color constants matching theme ─────────────────────────────
_GREEN = "#22c55e"
_YELLOW = "#eab308"
_ORANGE = "#f97316"
_RED = "#ef4444"
_BLUE = "#3b82f6"
_CYAN = "#06b6d4"
_MUTED = "#5a6a80"
_BG_CARD = "#1a2234"
_BG_SEC = "#111827"


def _risk_color(score: float) -> str:
    if score < 0.25:
        return _GREEN
    elif score < 0.5:
        return _YELLOW
    elif score < 0.75:
        return _ORANGE
    return _RED


def _risk_word(score: float) -> str:
    if score < 0.25:
        return "LOW"
    elif score < 0.5:
        return "MODERATE"
    elif score < 0.75:
        return "HIGH"
    return "CRITICAL"


def _status_class(score: float) -> str:
    if score < 0.25:
        return "status-safe"
    elif score < 0.5:
        return "status-caution"
    elif score < 0.75:
        return "status-warning"
    return "status-critical"


def render_metrics_header(summary: dict[str, Any]) -> None:
    risk = summary.get("latest_current_risk", 0)
    pred = summary.get("latest_predicted_risk", 0)
    risk_word = _risk_word(risk)
    risk_cls = _status_class(risk)

    # Status badge
    st.markdown(f'<span class="{risk_cls}">{risk_word}</span>', unsafe_allow_html=True)

    # Primary metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak People", int(summary["peak_people_count"]))
    c2.metric("Current Risk", f"{risk:.2f}")
    c3.metric("Predicted Risk", f"{pred:.2f}")
    c4.metric("Entered / Exited", f"{summary.get('entries_total', 0)} / {summary.get('exits_total', 0)}")

    # Secondary metrics
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Highest Risk", f"{summary.get('highest_current_risk', 0):.2f}")
    c6.metric("Bottleneck Risk", f"{summary.get('predicted_bottleneck_risk', 0):.2f}")
    c7.metric("Tracking Stability", f"{summary.get('tracking_stability', 1.0):.2f}")
    c8.metric("ID Switches", int(summary.get("id_switches", 0)))


def render_alerts(alerts: list[str]) -> None:
    st.subheader("Alerts")
    if not alerts:
        st.success("All zones within safe limits.")
        return
    for alert in alerts[-6:]:
        if "CRITICAL" in alert or "IMMEDIATE" in alert or "DEPLOY" in alert:
            st.error(alert)
        elif "BIMODAL" in alert or "escalation" in alert:
            st.warning(alert)
        else:
            st.info(alert)


def render_recommendations(recommendations: list[str]) -> None:
    st.subheader("Recommendations")
    if not recommendations:
        st.info("No actions required. Continue monitoring.")
        return
    for item in recommendations[:6]:
        st.markdown(f"→ {item}")


def render_operator_actions(actions: list[str]) -> None:
    """Operator action cards with semantic urgency styling."""
    st.subheader("Operator Actions")
    if not actions:
        st.markdown('<div class="action-clear">ALL CLEAR — All zones within safe limits</div>',
                    unsafe_allow_html=True)
        return
    for action in actions:
        if action.startswith("IMMEDIATE"):
            st.markdown(f'<div class="action-immediate">{action}</div>', unsafe_allow_html=True)
        elif action.startswith("PRE-EMPTIVE"):
            st.markdown(f'<div class="action-preemptive">{action}</div>', unsafe_allow_html=True)
        elif action.startswith("CAUTION"):
            st.markdown(f'<div class="action-preemptive">{action}</div>', unsafe_allow_html=True)
        elif action.startswith("WATCH"):
            st.markdown(f'<div class="action-watch">{action}</div>', unsafe_allow_html=True)
        elif action.startswith("ALL CLEAR"):
            st.markdown(f'<div class="action-clear">{action}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="action-watch">{action}</div>', unsafe_allow_html=True)


def render_zone_table(zone_snapshots: list[dict[str, Any]]) -> None:
    st.subheader("Zone Status")
    if not zone_snapshots:
        st.info("No zone data available.")
        return
    df = pd.DataFrame(zone_snapshots)
    if df.empty:
        return
    # Rename columns for operator readability
    col_map = {
        "zone": "Zone",
        "zone_type": "Type",
        "count": "People",
        "local_density": "Density",
        "local_risk": "Risk",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    display_cols = [c for c in ["Zone", "Type", "People", "Risk"] if c in df.columns]
    if "Risk" in df.columns:
        df = df.sort_values("Risk", ascending=False)
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def render_zone_predictions(zone_predictions: dict[str, dict[str, Any]]) -> None:
    """Zone prediction cards with visual urgency."""
    st.subheader("Zone Predictions")
    if not zone_predictions:
        st.info("Prediction data not available yet.")
        return

    for zname, zp in zone_predictions.items():
        trend = zp.get("risk_trend", "stable")
        ttc = zp.get("time_to_critical")
        current = zp.get("current_density", 0)
        predicted = zp.get("predicted_density", 0)
        net_flow = zp.get("net_flow", 0)
        src = zp.get("primary_inflow_source")

        # Determine urgency
        if trend == "rapidly_increasing" or (ttc is not None and ttc < 60):
            border_color = _RED
            trend_word = "SURGING"
        elif "increasing" in trend or (ttc is not None and ttc < 120):
            border_color = _ORANGE
            trend_word = "Rising"
        elif "decreasing" in trend:
            border_color = _CYAN
            trend_word = "Easing"
        else:
            border_color = _GREEN
            trend_word = "Steady"

        # Time to critical in human language
        if ttc is not None and ttc < 60:
            ttc_text = f"⚠ Critical in {ttc:.0f}s"
            ttc_color = _RED
        elif ttc is not None and ttc < 180:
            ttc_text = f"~{ttc:.0f}s to limit"
            ttc_color = _ORANGE
        elif ttc is not None:
            ttc_text = f"~{ttc / 60:.0f}min to limit"
            ttc_color = _MUTED
        else:
            ttc_text = "Within limits"
            ttc_color = _GREEN

        # Render card
        st.markdown(
            f"""<div class="zone-card" style="border-left: 3px solid {border_color};">
            <div class="zone-card-header">
                <span class="zone-card-name">{zname}</span>
                <span style="color:{border_color}; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:0.8rem;">{trend_word}</span>
            </div>
            <div style="display:flex; gap:24px; flex-wrap:wrap;">
                <span class="zone-card-stat">Now <strong>{current:.0f}</strong></span>
                <span class="zone-card-stat">Predicted <strong>{predicted:.0f}</strong></span>
                <span class="zone-card-stat">Flow <strong>{net_flow:+.1f}</strong></span>
                <span class="zone-card-stat" style="color:{ttc_color};">{ttc_text}</span>
            </div>
            {"<div style='margin-top:6px; font-size:0.75rem; color:" + _MUTED + ";'>Inflow from " + src + "</div>" if src and zp.get("inflow_pressure", 0) > 0 else ""}
            </div>""",
            unsafe_allow_html=True
        )


def render_trend_charts(timeline_df: pd.DataFrame) -> None:
    st.subheader("Trend Analysis")
    if timeline_df.empty:
        st.info("No trend data available.")
        return

    chart_template = "plotly_dark"
    chart_colors = [_CYAN, _ORANGE, _RED, _GREEN]
    chart_height = 280

    layout_overrides = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        font=dict(family="Inter, sans-serif", size=11, color="#8896ab"),
        margin=dict(l=40, r=20, t=36, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(42,52,72,0.5)", zerolinecolor="rgba(42,52,72,0.5)"),
        yaxis=dict(gridcolor="rgba(42,52,72,0.5)", zerolinecolor="rgba(42,52,72,0.5)"),
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=timeline_df["second"], y=timeline_df["people_count"],
                                   name="People", line=dict(color=_CYAN, width=2)))
        fig1.add_trace(go.Scatter(x=timeline_df["second"], y=timeline_df["avg_speed"],
                                   name="Avg Speed", line=dict(color=_ORANGE, width=1.5, dash="dot")))
        fig1.update_layout(title="People & Movement", height=chart_height, template=chart_template, **layout_overrides)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timeline_df["second"], y=timeline_df["current_risk"],
                                   name="Current Risk", line=dict(color=_CYAN, width=2),
                                   fill="tozeroy", fillcolor="rgba(6,182,212,0.08)"))
        fig2.add_trace(go.Scatter(x=timeline_df["second"], y=timeline_df["predicted_risk"],
                                   name="Predicted Risk", line=dict(color=_RED, width=1.5, dash="dash")))
        fig2.update_layout(title="Risk Timeline", height=chart_height, template=chart_template, **layout_overrides)
        st.plotly_chart(fig2, use_container_width=True)

    # Zone congestion chart
    risk_cols = ["max_zone_density", "predicted_bottleneck_risk"]
    if "bimodal_predicted_risk" in timeline_df.columns:
        risk_cols.append("bimodal_predicted_risk")

    fig3 = go.Figure()
    colors = [_YELLOW, _RED, _ORANGE]
    names = ["Zone Density", "Bottleneck Risk", "Bi-Modal Prediction"]
    for i, col in enumerate(risk_cols):
        if col in timeline_df.columns:
            fig3.add_trace(go.Scatter(x=timeline_df["second"], y=timeline_df[col],
                                       name=names[i] if i < len(names) else col,
                                       line=dict(color=colors[i] if i < len(colors) else _MUTED, width=1.5)))
    fig3.update_layout(title="Zone Congestion & Prediction", height=chart_height, template=chart_template, **layout_overrides)
    st.plotly_chart(fig3, use_container_width=True)