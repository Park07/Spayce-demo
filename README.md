# Crowd Stress Detector

Crowd Stress Detector is a hackathon MVP for proactive venue safety. It analyzes crowd video, tracks movement, estimates congestion risk, and surfaces explainable alerts before dangerous crowd pressure escalates.

## Why this exists

Most crowd incidents are preceded by warning signals: rising density, flow conflicts, and localized bottlenecks near exits. This project demonstrates a startup-style operations dashboard that turns those signals into practical interventions for safety teams.

## Core features

- Upload and process crowd video in a Streamlit dashboard
- YOLOv8 person detection with ID-based movement tracking
- Heatmap overlay of crowd density hotspots
- Zone-based analysis for exits, entries, and bottlenecks
- Explainable `Crowd Stress Risk` scoring (`0.0` to `1.0`)
- Short-horizon risk prediction (next `10-20s`)
- Anomaly detection:
  - sudden density spikes
  - directional conflict
  - stopped flow near exits
  - clustering near bottlenecks
- Alerts and operator recommendations
- Trend charts (count, speed, risk, predicted risk, zone pressure)
- PDF-style session report export
- Optional mocked split-screen multi-camera mode

## Tech stack

- Frontend: Streamlit
- CV backend: Python, OpenCV, Ultralytics YOLOv8
- Data & analytics: NumPy, Pandas
- Visualization: Plotly
- Report export: ReportLab

## Architecture

The project is modular so a 4-person team can work in parallel:

- **Member 1:** `core/detector.py`, `core/tracker.py`
- **Member 2:** `core/heatmap.py`, `core/zones.py`, `core/metrics.py`
- **Member 3:** `core/risk.py`, `core/anomaly.py`, `core/recommendations.py`
- **Member 4:** `app/main.py`, `app/ui.py`, `core/export.py`, dashboard UX

## Project structure

```text
crowd-stress-detector/
├── app/
│   ├── main.py
│   ├── ui.py
│   └── theme.py
├── core/
│   ├── detector.py
│   ├── tracker.py
│   ├── heatmap.py
│   ├── metrics.py
│   ├── risk.py
│   ├── anomaly.py
│   ├── zones.py
│   ├── recommendations.py
│   ├── video_processor.py
│   ├── export.py
│   ├── config.py
│   └── utils.py
├── data/
│   └── sample_videos/
├── outputs/
│   ├── reports/
│   └── processed/
├── tests/
│   ├── test_risk.py
│   └── test_zones.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app/main.py
```

4. Open the Streamlit URL shown in your terminal.

## How to use

1. Upload a crowd video (`.mp4`, `.mov`, `.avi`, `.mkv`)
2. Enable optional features:
   - heatmap overlay
   - zone analysis
   - trend charts
3. Process the session
4. Review:
   - processed video
   - current and predicted Crowd Stress Risk
   - anomalies, alerts, and recommendations
   - trend graphs and zone table
5. Export PDF-style summary

## Configurable behavior

All scoring weights, thresholds, and default zones live in `core/config.py`:

- risk weighting
- anomaly thresholds
- density assumptions
- zone definitions

## Future improvements

- webcam / RTSP live stream ingestion
- stronger forecasting model beyond heuristics
- cross-camera identity stitching
- calibrated floor-plane density estimation
- notification integrations (Slack/SMS)
- operator action logging and feedback loop

## Notes for demo stability

- Runs on CPU-only laptops
- Handles empty frames and zero-person scenes safely
- Uses explainable heuristics for predictable hackathon demos
