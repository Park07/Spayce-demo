from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas


def build_session_pdf(session: dict[str, Any]) -> bytes:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 48

    summary = session["summary"]
    anomalies = session["anomalies"]
    alerts = session["alerts"]
    recommendations = session["recommendations"]

    c.setFont("Helvetica-Bold", 18)
    c.drawString(48, y, "Crowd Stress Detector - Session Summary")
    y -= 30
    c.setFont("Helvetica", 11)
    lines = [
        f"Session ID: {summary['session_id']}",
        f"Processed Video: {summary['video_name']}",
        f"Peak People Count: {summary['peak_people_count']}",
        f"Highest Current Risk: {summary['highest_current_risk']:.2f}",
        f"Highest Predicted Risk: {summary['highest_predicted_risk']:.2f}",
        f"Predicted Bottleneck Risk: {summary.get('predicted_bottleneck_risk', 0.0):.2f}",
        f"Final Risk Level: {summary['risk_level']}",
        f"Busiest Zone: {summary['busiest_zone']}",
        f"Total Entering: {summary.get('entries_total', 0)}",
        f"Total Exiting: {summary.get('exits_total', 0)}",
        f"Dominant Direction: {summary.get('dominant_direction', 'UNKNOWN')}",
        f"Active Tracks: {summary.get('active_tracks', 0)}",
        f"ID Switches: {summary.get('id_switches', 0)}",
        f"Lost Tracks: {summary.get('lost_tracks', 0)}",
        f"Recovered Tracks: {summary.get('recovered_tracks', 0)}",
        f"Tracking Stability: {summary.get('tracking_stability', 1.0):.2f}",
        f"Anomaly Count: {len(anomalies)}",
    ]
    for line in lines:
        c.drawString(48, y, line)
        y -= 18

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "Top Alerts")
    y -= 18
    c.setFont("Helvetica", 11)
    for alert in alerts[:6]:
        for wrapped in simpleSplit(f"- {alert}", "Helvetica", 11, width - 90):
            c.drawString(54, y, wrapped)
            y -= 16

    y -= 8
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, y, "Operator Recommendations")
    y -= 18
    c.setFont("Helvetica", 11)
    for rec in recommendations[:6]:
        for wrapped in simpleSplit(f"- {rec}", "Helvetica", 11, width - 90):
            c.drawString(54, y, wrapped)
            y -= 16

    c.save()
    return buf.getvalue()
