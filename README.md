# Spayce — Predictive Crowd Stress Detection

**Spayce** transforms existing CCTV feeds into real-time crowd risk intelligence. It detects people, tracks movement, predicts where congestion will build, and tells operators what to do — before a situation becomes dangerous.

Built at FoundersHack Sydney 2026. Extended with BINTS neural network integration post-hackathon.

---

## The Problem

159 people died at Itaewon in 2022. Overcrowding calls started at 6pm — nobody acted until 10pm. The gap wasn't awareness. It was connecting the warning to an action.

The NSW Auditor-General found Sydney Trains rates platform overcrowding as a **high strategic risk** but has **no strategy to manage it**. Existing tools measure crowds after the fact. Nobody predicts what happens next or tells operators what to do about it.

## The Solution

Spayce operates across three layers:

| Layer | What it does | Technology |
|-------|-------------|------------|
| **Detection** | Counts and tracks every person in frame | P2PNet (ICCV 2021) |
| **Prediction** | Forecasts density per zone using density + flow signals | BINTS-adapted TCN + GCN (KDD 2025) |
| **Execution** | Generates operator actions when thresholds are approached | Rule-based action engine |

---

## Architecture

```
Video Frame
    │
    ▼
┌──────────────┐
│   P2PNet     │  Head point detection (70-80 detections/frame)
│   Detection  │  Nearest-neighbor ID tracking
└──────┬───────┘
       │  tracked_people: [{track_id, centroid, trajectory, confidence}, ...]
       ▼
┌──────────────┐
│  Zone Grid   │  2×3 grid: 6 zones with adjacency
│  Assignment  │  People assigned to zones by centroid position
└──────┬───────┘
       │  density per zone, flow between zones
       ▼
┌──────────────────────────────────────────┐
│         BINTS Prediction Layer           │
│                                          │
│  ┌─────────┐    ┌─────────┐             │
│  │  TCN    │    │  GCN    │             │
│  │(temporal)│    │(spatial)│             │
│  └────┬────┘    └────┬────┘             │
│       │              │                   │
│       ▼              ▼                   │
│  ┌─────────────────────┐                │
│  │ Cosine Similarity   │  Bi-modal      │
│  │ Fusion              │  alignment     │
│  └──────────┬──────────┘                │
│             ▼                            │
│  ┌─────────────────────┐                │
│  │ Feature Extractor   │                │
│  │ + Linear Predictor  │                │
│  └──────────┬──────────┘                │
│             ▼                            │
│  predicted_density, risk_trend, TTC      │
│  + EMA baseline (blended output)         │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Execution   │  Operator actions: close entry, deploy security,
│  Engine      │  PA announcement, open alternate routes
└──────────────┘
```

---

## Detection Layer

**Model:** [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet) (ICCV 2021, Tencent)

P2PNet detects head points directly rather than full-body bounding boxes. In dense crowds, bodies overlap but heads remain visible from elevated camera angles. This makes P2PNet significantly more accurate than general-purpose detectors like YOLO in crowd scenarios.

**Detection evolution during development:**

| Approach | Detections | Issue |
|----------|-----------|-------|
| YOLOv8 (COCO) | 12 | Overlapping bodies merged |
| YOLOv8-Face + SAHI | 41 | Oscillated 20-80, unstable |
| **P2PNet** | **70-80** | **Stable, purpose-built** |

**Tracking:** Nearest-neighbor ID matching with 50px max distance threshold. Maintains 30-frame trajectory history per person. Tracking stability: 0.94-1.00.

---

## Prediction Layer

**Based on:** [BINTS: Bi-Modal Learning for Networked Time Series](https://doi.org/10.1145/3711896.3736856) (KDD 2025, KAIST)

**Core insight:** Crowd prediction requires two signals working together — **density** (how many people per zone) and **flow** (how people move between zones). BINTS proved this bi-modal approach is up to 76% more accurate than single-modality methods.

### Components (imported from BINTS codebase)

| Component | Source | Purpose |
|-----------|--------|---------|
| `TemporalConvNet` | `BINTS/models.py` | Learns temporal patterns in density history |
| `pairwise_cosine_sim` | `BINTS/calculator.py` | Bi-modal fusion between TCN and GCN outputs |
| `TemporalContrastiveLoss` | `BINTS/contrastive_losses.py` | Enforces smooth temporal representations |

### Our additions

| Component | File | Purpose |
|-----------|------|---------|
| `SimpleGCN` | `core/prediction.py` | Spatial relationship learning via graph convolution |
| `OnlineBINTSModel` | `core/prediction.py` | Wires TCN + GCN + fusion + predictor |
| Online training loop | `core/prediction.py` | Trains every 5 frames on accumulated data |
| EMA blending | `core/prediction.py` | Blends NN output with statistical baseline |

### Adaptation from BINTS

| BINTS (original) | Spayce (adapted) |
|-------------------|------------------|
| 233 subway stations | 6 camera zones (2×3 grid) |
| 274 days historical data | Online learning from scratch |
| Offline batch training on GPU | Online incremental training on CPU |
| Hourly time resolution | Frame-rate resolution (~0.2s) |
| Full GCN with torch_geometric | SimpleGCN with PyG or fallback |

### Training

The model trains online as it processes video:

1. **Frames 1-19:** Accumulate data. EMA baseline only.
2. **Frame 20:** First training step. Creates sliding windows from history, runs forward pass, computes loss, backpropagates.
3. **Every 5 frames after:** Additional training step with growing data.
4. **Blended output:** NN prediction weighted by inverse loss — high loss = EMA dominates, low loss = NN gets up to 50% weight.

**Loss function:** `0.5 × prediction_MSE + 0.01 × temporal_contrastive_loss`

**Results on Tokyo crosswalk (11 seconds, 550 frames):**

| Metric | Value |
|--------|-------|
| Training loss (start → end) | 3200 → 1.04 |
| EMA prediction error | 1-5 people |
| NN prediction range | In correct range after normalization |
| Training steps fired | 4-6 per video |

---

## Zone Configuration

The frame is divided into a 2×3 grid:

```
┌──────────┬──────────┬──────────┐
│ Zone_TL  │ Zone_TC  │ Zone_TR  │
│ (entry)  │(bottlenk)│  (exit)  │
├──────────┼──────────┼──────────┤
│ Zone_BL  │ Zone_BC  │ Zone_BR  │
│ (entry)  │(bottlenk)│  (exit)  │
└──────────┴──────────┴──────────┘
```

**Adjacency:** Each zone connects to its horizontal and vertical neighbors (7 edges). This gives the GCN real spatial structure to learn from, and creates meaningful flow signals when people move between zones.

For production deployment, zones would be configured per venue — matching actual entry gates, corridors, exits, and chokepoints.

---

## Execution Layer

When predicted density crosses a threshold, the system generates operator actions:

- **IMMEDIATE: Close entry** — when zone exceeds capacity
- **IMMEDIATE: Deploy security** — when zone is over capacity
- **CAUTION: Approaching capacity** — when zone reaches 80% of limit
- **Recommendation: Open alternate route** — when bottleneck detected
- **Recommendation: Pre-position response team** — predictive action
- **BIMODAL alert** — when density-flow divergence detected

Actions are validated against real crowd management protocols confirmed by security professionals.

---

## Setup

### Requirements

- Python 3.11+
- PyTorch 2.0+
- torch_geometric
- OpenCV
- Streamlit

### Installation

```bash
git clone https://github.com/Park07/Spayce-demo.git
cd Spayce-demo/crowd-stress-detector

# Install dependencies
pip install -r requirements.txt
pip install torch_geometric

# Clone P2PNet (for detection)
cd ~
git clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet.git
```

### Running

```bash
cd crowd-stress-detector
streamlit run app/main.py
```

Upload a crowd video. Adjust settings in the sidebar:
- **Zone capacity limit:** People per zone before alerting
- **Detection confidence:** P2PNet confidence threshold (0.90+ recommended)
- **Frame stride:** Process every Nth frame (higher = faster, fewer frames)

---

## Project Structure

```
crowd-stress-detector/
├── app/
│   ├── main.py              # Streamlit dashboard
│   ├── ui.py                # UI components
│   └── theme.py             # Dark theme styling
├── core/
│   ├── prediction.py        # BINTS-adapted bi-modal predictor
│   ├── p2pnet_detector.py   # P2PNet detection wrapper
│   ├── video_processor.py   # Main processing pipeline
│   ├── config.py            # Zone grid, thresholds, weights
│   ├── risk.py              # Risk scoring
│   ├── metrics.py           # Density, speed, clustering
│   ├── zones.py             # Zone management and rendering
│   ├── anomaly.py           # Anomaly detection
│   ├── recommendations.py   # Operator action generation
│   ├── heatmap.py           # Density heatmap overlay
│   └── export.py            # PDF report generation
├── BINTS/                   # Cloned BINTS repository (KAIST)
│   ├── models.py            # TCN, GCN, FeatureExtractor, ResNet1D
│   ├── BINTS.py             # Original CLIP_3D training loop
│   ├── contrastive_losses.py # Ranking + Temporal contrastive losses
│   ├── calculator.py        # Cosine similarity
│   ├── data_loader.py       # Seoul/Busan/Daegu dataset loading
│   └── main.py              # Original entry point
├── outputs/
│   └── logs/                # Per-run frame-by-frame CSV logs
├── test_prediction.py       # Prediction validation script
└── requirements.txt
```

---

## Research References

| Paper | Conference | Used For |
|-------|-----------|----------|
| [BINTS: Bi-Modal Learning for Networked Time Series](https://doi.org/10.1145/3711896.3736856) | KDD 2025 | TCN, cosine similarity, contrastive loss |
| [P2PNet: Rethinking Counting and Localization in Crowds](https://arxiv.org/abs/2107.12746) | ICCV 2021 | Head point detection |
| [Itaewon crowd crush analysis](https://doi.org/10.1016/j.ssci.2024.106485) | ScienceDirect 2025 | Density thresholds |
| Lee & Hughes 2006 | — | Crowd pressure = density × variance |
| Keith Still PhD research | — | Critical density thresholds |

---

## Customer Validation

- **Andrew Tatrai, PhD** — Founder of DCM (Dynamic Crowd Measurement) and ACESGroup. 46 years in crowd management. Winner of 2025 GSIC × Microsoft Sports-Tech Innovation Challenge. Met in person to discuss technology gap and potential collaboration.
- **Security professionals** — Validated operator actions match real crowd crush response protocols.
- **Commuter interviews** — 4 interviews at Sydney transit stations confirming the need for predictive warnings.
- **NSW Auditor-General Report (2020)** — Confirmed Sydney Trains has no strategy for platform overcrowding despite rating it as high strategic risk.

---

## Known Limitations

- **CPU processing speed:** ~5 seconds per frame on M2 Mac. GPU required for real-time deployment.
- **Neural network accuracy:** On short videos (11 seconds), the statistical EMA baseline outperforms the neural network. The NN needs hours of venue-specific data to learn meaningful patterns.
- **Single camera:** Currently processes one video feed. Multi-camera support requires backend infrastructure.
- **Fixed zone grid:** The 2×3 grid is a default. Production deployment needs per-venue zone configuration.
- **No SMS dispatch yet:** Twilio account configured but not integrated.
- **Liability framing:** Prediction systems carry liability risk if predictions are wrong. System should be positioned as decision-support, not autonomous safety.

---

## Team

- **William Park** — Detection pipeline, prediction layer, BINTS integration, customer validation
- **Bhakthi Salimath** — UI/UX, Streamlit dashboard, initial MVP
- **Joshua Yee** — Execution layer research
- **Tushti Chaturvedi** — Market research, DCM analysis

Built at FoundersHack Sydney 2026. Extended March 21-23, 2026.

---

## License

Research prototype. BINTS code is from [github.com/kaist-dmlab/BINTS](https://github.com/kaist-dmlab/BINTS) under their original license. P2PNet from [TencentYoutuResearch](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet).