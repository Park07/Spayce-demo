"""
Bi-Modal Crowd Prediction Layer
================================
Adapted from BINTS: Bi-Modal Learning for Networked Time Series (KDD 2025)
Paper: https://doi.org/10.1145/3711896.3736856
Authors: Nam et al., KAIST
Code: https://github.com/kaist-dmlab/BINTS

This module adapts BINTS for online crowd prediction from video:
- Uses BINTS's TemporalConvNet (TCN) for temporal pattern learning
- Uses BINTS's pairwise_cosine_sim for bi-modal fusion
- Uses BINTS's TemporalContrastiveLoss for temporal contrastive learning
- Replaces batch training with online incremental training
- Uses GCNConv for spatial relationship learning between zones

Key adaptation: BINTS trains offline on 274 days of hourly transit data.
We train online on rolling video frames. The model improves as it processes
more frames, starting from random initialization.

Research-backed thresholds from:
- Itaewon stampede analysis (ScienceDirect 2025)
- Keith Still PhD (crowd safety pioneer)
- PLOS ONE 2021 entropy model
- Lee & Hughes 2006 (crowd pressure = density × variance)
"""

from __future__ import annotations

import sys
import os
import math
import numpy as np
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import BINTS components directly from the cloned repository
import importlib.util

BINTS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BINTS")

def _load_bints_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BINTS_ROOT, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_models = _load_bints_module("models")
_calculator = _load_bints_module("calculator")
_contrastive = _load_bints_module("contrastive_losses")

TemporalConvNet = _models.TemporalConvNet
pairwise_cosine_sim = _calculator.pairwise_cosine_sim
TemporalContrastiveLoss = _contrastive.TemporalContrastiveLoss
from core.config import ZONE_ADJACENCY, PREDICTION_WEIGHTS


# ─── Lightweight GCN (avoids full torch_geometric if needed) ────────
class SimpleGCN(nn.Module):
    """
    Simplified Graph Convolutional layer.
    Adapted from BINTS's GraphNet but without torch_geometric dependency fallback.
    If torch_geometric is available, uses GCNConv. Otherwise, manual adjacency multiplication.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
            self.use_pyg = True
        except ImportError:
            # Fallback: manual graph convolution via adjacency matrix multiplication
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.use_pyg = False
        self.output = nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index=None, adj_matrix=None):
        """
        Args:
            x: [batch, num_nodes, features]
            edge_index: [2, num_edges] for PyG
            adj_matrix: [num_nodes, num_nodes] for fallback
        """
        batch_size, num_nodes, num_features = x.size()

        if self.use_pyg and edge_index is not None:
            x_flat = x.reshape(-1, num_features)
            x_flat = F.relu(self.conv1(x_flat, edge_index))
            x_flat = F.relu(self.conv2(x_flat, edge_index))
            x_flat = self.output(x_flat)
            return x_flat.reshape(batch_size, num_nodes, -1)
        else:
            # Manual GCN: A @ X @ W
            if adj_matrix is not None:
                # Normalize adjacency: D^(-1/2) A D^(-1/2)
                deg = adj_matrix.sum(dim=1).clamp(min=1)
                deg_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg))
                adj_norm = deg_inv_sqrt @ adj_matrix @ deg_inv_sqrt
                x = torch.bmm(adj_norm.unsqueeze(0).expand(batch_size, -1, -1), x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.output(x)


# ─── Online BINTS Prediction Model ─────────────────────────────────
class OnlineBINTSModel(nn.Module):
    """
    Online adaptation of BINTS architecture.

    Components (mapped to BINTS paper):
    - TCN: processes density time series per zone (Section 4.3, node-oriented)
    - GCN: processes flow between zones (Section 4.3, edge-oriented)
    - Cosine similarity: bi-modal fusion (Section 4.4, Eq. 3)
    - Linear predictor: forecasts future density (Section 4.6)
    """
    def __init__(self, num_zones, seq_len=30, pred_len=5):
        super().__init__()
        self.num_zones = num_zones
        self.seq_len = seq_len
        self.pred_len = pred_len

        # TCN for temporal density patterns (BINTS Section 4.3)
        # Input: density per zone over time → temporal representation
        tcn_channels = [16, 32, num_zones]
        self.tcn = TemporalConvNet(
            num_inputs=num_zones,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=0.2,
        )

        # GCN for spatial flow patterns (BINTS Section 4.3)
        # Input: OD flow matrix → spatial representation
        self.gcn = SimpleGCN(
            input_dim=seq_len,
            hidden_dim=32,
            output_dim=num_zones,
        )

        # Feature extractor: processes cosine similarity matrix (BINTS Section 4.4)
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_zones * num_zones, 64),
            nn.ReLU(),
            nn.Linear(64, num_zones * seq_len),
        )

        # Prediction head (BINTS Section 4.6)
        # Input: original signal + extracted features → predicted density
        self.predictor = nn.Linear(seq_len, pred_len)
        self.gcn_proj = nn.Linear(num_zones, seq_len)

    def forward(self, density_seq, flow_seq, adj_matrix=None, edge_index=None):
        """
        Args:
            density_seq: [batch, seq_len, num_zones] — density per zone over time
            flow_seq: [batch, num_zones, seq_len] — flow features per zone over time
            adj_matrix: [num_zones, num_zones] — zone adjacency
            edge_index: [2, num_edges] — for PyG GCN

        Returns:
            prediction: [batch, pred_len, num_zones] — predicted density
            tcn_repr: temporal representation (for contrastive loss)
            gcn_repr: spatial representation (for contrastive loss)
        """
        batch_size = density_seq.size(0)

        # TCN: process density time series
        # Input shape for TCN: [batch, channels, seq_len]
        tcn_input = density_seq.permute(0, 2, 1)  # [batch, num_zones, seq_len]
        tcn_out = self.tcn(tcn_input)  # [batch, num_zones, seq_len]

        # GCN: process flow features
        # Input: [batch, num_zones, seq_len]
        gcn_out = self.gcn(flow_seq, edge_index=edge_index, adj_matrix=adj_matrix)
        # gcn_out: [batch, num_zones, num_zones]

        # Bi-modal fusion via cosine similarity (BINTS Eq. 3)
        # tcn_repr: [batch, num_zones, seq_len]
        # gcn_repr: [batch, num_zones, num_zones] → pad/project to match
        tcn_repr = tcn_out
        gcn_repr = gcn_out

        # Cosine similarity between temporal and spatial representations
        # Need same last dimension — project gcn to seq_len
        gcn_repr_proj = self.gcn_proj(gcn_repr)

        cos_sim = pairwise_cosine_sim((tcn_repr, gcn_repr_proj))
        # cos_sim: [batch, num_zones, num_zones]

        # Feature extraction from cosine similarity
        cos_flat = cos_sim.reshape(batch_size, -1)  # [batch, num_zones²]
        features = self.feature_extractor(cos_flat)  # [batch, num_zones * seq_len]
        features = features.reshape(batch_size, self.seq_len, self.num_zones)

        # Combine original input with extracted features (BINTS Section 4.6)
        combined = density_seq + features  # residual connection

        # Predict future density
        # combined: [batch, seq_len, num_zones]
        combined_t = combined.permute(0, 2, 1)  # [batch, num_zones, seq_len]
        prediction = self.predictor(combined_t)  # [batch, num_zones, pred_len]
        prediction = prediction.permute(0, 2, 1)  # [batch, pred_len, num_zones]

        return prediction, tcn_repr, gcn_repr_proj


# ─── Main Predictor Class ──────────────────────────────────────────
class BiModalPredictor:
    """
    BINTS-adapted bi-modal predictor with online training.

    Combines:
    - Zone density tracking (node-oriented signal)
    - Zone-to-zone flow tracking (edge-oriented signal)
    - TCN for temporal pattern learning
    - GCN for spatial relationship learning
    - Online training every N frames
    - EMA fallback when model hasn't trained enough
    """

    def __init__(
        self,
        zone_names: list[str],
        zone_types: dict[str, str] | None = None,
        window_size: int = 90,
        prediction_horizon: int = 5,
        gaussian_sigma: float = 10.0,
        critical_density_per_zone: float = 20.0,
    ):
        self.zone_names = zone_names
        self.zone_types = zone_types or {}
        self.num_zones = max(len(zone_names), 1)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.gaussian_sigma = gaussian_sigma
        self.critical_density = critical_density_per_zone

        # Data buffers
        self.density_history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.flow_history: deque[dict[tuple[str, str], float]] = deque(maxlen=window_size)
        self.speed_history: deque[dict[str, float]] = deque(maxlen=window_size)
        self.prev_zone_assignments: dict[int, str] = {}

        # EMA smoothing (keeps working as fallback)
        self.ema_density: dict[str, float] = {}
        self.ema_alpha = 0.3

        # ─── BINTS Neural Network Components ───
        self.device = torch.device('cpu')
        self.seq_len = 15  # lookback window for TCN
        self.pred_len = prediction_horizon

        # Build model
        self.model = OnlineBINTSModel(
            num_zones=self.num_zones,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
        ).to(self.device)

        # Optimizer (BINTS uses SGD with momentum for TCN/GCN, Adam for predictor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)

        # Losses (from BINTS)
        self.prediction_criterion = nn.MSELoss()
        self.temporal_criterion = TemporalContrastiveLoss(
            temperature=0.2, softplus_w=1.0  # reduced from 10e3 for online stability
        )

        # Training state
        self.frame_count = 0
        self.train_every_n = 5  # train every 30 frames
        self.min_train_frames = 20  # minimum frames before first training
        self.is_trained = False
        self.training_losses = []

        # Build adjacency matrix for GCN
        self.adj_matrix = self._build_adjacency_matrix()
        self.edge_index = self._build_edge_index()

    def _build_adjacency_matrix(self) -> torch.Tensor:
        """Build adjacency matrix from zone connectivity."""
        adj = torch.eye(self.num_zones)
        for i, zi in enumerate(self.zone_names):
            for j, zj in enumerate(self.zone_names):
                if i != j:
                    weight = ZONE_ADJACENCY.get((zi, zj), 0.0)
                    if weight == 0:
                        weight = ZONE_ADJACENCY.get((zj, zi), 0.0)
                    adj[i, j] = weight
        # If single zone or no adjacency defined, create fully connected
        if adj.sum() <= self.num_zones:  # only self-loops
            adj = torch.ones(self.num_zones, self.num_zones)
        return adj.to(self.device)

    def _build_edge_index(self) -> torch.Tensor:
        """Build edge index from adjacency matrix for PyG GCNConv."""
        edges = self.adj_matrix.nonzero(as_tuple=False).t()
        return edges.long().to(self.device)

    def _assign_zones(
        self, tracked_people: list[dict[str, Any]], zones: list[dict[str, Any]]
    ) -> dict[int, str]:
        assignments = {}
        for person in tracked_people:
            tid = person.get("track_id", -1)
            if tid < 0:
                continue
            cx, cy = person["centroid"]
            for zone in zones:
                zname = zone.get("name", zone.get("zone", ""))
                rect = zone["rect"]
                x1, y1, x2, y2 = rect
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    assignments[tid] = zname
                    break
        return assignments

    def _compute_density(self, assignments: dict[int, str]) -> dict[str, float]:
        density = {z: 0.0 for z in self.zone_names}
        for zone in assignments.values():
            if zone in density:
                density[zone] += 1.0
        return density

    def _compute_flow(self, current_assignments: dict[int, str]) -> dict[tuple[str, str], float]:
        flow: dict[tuple[str, str], float] = {}
        for tid, current_zone in current_assignments.items():
            prev_zone = self.prev_zone_assignments.get(tid)
            if prev_zone is not None and prev_zone != current_zone:
                key = (prev_zone, current_zone)
                flow[key] = flow.get(key, 0.0) + 1.0
        return flow

    def _compute_zone_speeds(
        self, tracked_people: list[dict[str, Any]], assignments: dict[int, str]
    ) -> dict[str, float]:
        zone_speeds: dict[str, list[float]] = {z: [] for z in self.zone_names}
        for person in tracked_people:
            tid = person.get("track_id", -1)
            if tid < 0:
                continue
            zone = assignments.get(tid)
            if zone is None:
                continue
            trajectory = person.get("trajectory", [])
            if len(trajectory) >= 2:
                (x1, y1), (x2, y2) = trajectory[-2], trajectory[-1]
                speed = math.hypot(x2 - x1, y2 - y1)
                zone_speeds[zone].append(speed)
        return {z: float(np.mean(speeds)) if speeds else 0.0 for z, speeds in zone_speeds.items()}

    def _update_ema(self, density: dict[str, float]) -> dict[str, float]:
        smoothed = {}
        for zone, raw_count in density.items():
            if zone not in self.ema_density:
                self.ema_density[zone] = raw_count
            else:
                self.ema_density[zone] = (
                    self.ema_alpha * raw_count
                    + (1 - self.ema_alpha) * self.ema_density[zone]
                )
            smoothed[zone] = self.ema_density[zone]
        return smoothed

    def _prepare_tensors(self):
        """
        Convert density and flow history buffers into tensors for the model.

        Returns:
            density_tensor: [1, seq_len, num_zones]
            flow_tensor: [1, num_zones, seq_len]
        """
        history_len = len(self.density_history)
        seq_len = min(history_len, self.seq_len)

        # Density tensor: per-zone counts over time
        density_data = []
        for d in list(self.density_history)[-seq_len:]:
            row = [d.get(z, 0.0) for z in self.zone_names]
            density_data.append(row)

        # Pad if shorter than seq_len
        while len(density_data) < self.seq_len:
            density_data.insert(0, density_data[0] if density_data else [0.0] * self.num_zones)

        density_tensor = torch.tensor([density_data], dtype=torch.float32, device=self.device)
        # density_tensor: [1, seq_len, num_zones]

        # Flow tensor: per-zone flow features over time
        flow_data = []
        for f in list(self.flow_history)[-seq_len:]:
            # Build per-zone inflow vector
            zone_flow = [0.0] * self.num_zones
            for (src, dst), count in f.items():
                if dst in self.zone_names:
                    idx = self.zone_names.index(dst)
                    zone_flow[idx] += count
            flow_data.append(zone_flow)

        while len(flow_data) < self.seq_len:
            flow_data.insert(0, [0.0] * self.num_zones)

        flow_tensor = torch.tensor([flow_data], dtype=torch.float32, device=self.device)
        flow_tensor = flow_tensor.permute(0, 2, 1)  # [1, num_zones, seq_len]

        return density_tensor, flow_tensor

    def _train_step(self):
        """
        Online training step using accumulated data.
        Called every train_every_n frames.

        Loss = 0.5 * prediction_loss + 0.25 * temporal_contrastive_loss
        (BINTS Eq. 11, without spatial contrastive since we have few zones)
        """
        if len(self.density_history) < self.seq_len + self.pred_len:
            return

        self.model.train()

        # Prepare input and target
        # Input: frames [0..seq_len-1], Target: frames [seq_len..seq_len+pred_len-1]
        all_density = []
        for d in list(self.density_history):
            row = [d.get(z, 0.0) for z in self.zone_names]
            all_density.append(row)

        all_density_np = np.array(all_density)

        # Create multiple training windows from the accumulated data
        num_windows = len(all_density) - self.seq_len - self.pred_len + 1
        if num_windows <= 0:
            return

        # Use last few windows for training (online learning)
        max_windows = min(num_windows, 4)  # batch of up to 8 windows
        inputs = []
        targets = []
        for i in range(num_windows - max_windows, num_windows):
            inp = all_density_np[i:i + self.seq_len]
            tgt = all_density_np[i + self.seq_len:i + self.seq_len + self.pred_len]
            inputs.append(inp)
            targets.append(tgt)

        density_input = torch.tensor(np.array(inputs), dtype=torch.float32, device=self.device)
        density_target = torch.tensor(np.array(targets), dtype=torch.float32, device=self.device)

        # Build flow input for the same windows
        all_flow = []
        for f in list(self.flow_history):
            zone_flow = [0.0] * self.num_zones
            for (src, dst), count in f.items():
                if dst in self.zone_names:
                    idx = self.zone_names.index(dst)
                    zone_flow[idx] += count
            all_flow.append(zone_flow)

        # Pad flow to match density length
        while len(all_flow) < len(all_density):
            all_flow.append([0.0] * self.num_zones)
        all_flow_np = np.array(all_flow[:len(all_density)])

        flow_inputs = []
        for i in range(num_windows - max_windows, num_windows):
            fl = all_flow_np[i:i + self.seq_len]
            flow_inputs.append(fl)

        flow_input = torch.tensor(np.array(flow_inputs), dtype=torch.float32, device=self.device)
        flow_input = flow_input.permute(0, 2, 1)  # [batch, num_zones, seq_len]

        # Normalize for stable training
        self._density_mean = density_input.mean()
        self._density_std = density_input.std().clamp(min=1.0)
        density_input = (density_input - self._density_mean) / self._density_std
        density_target = (density_target - self._density_mean) / self._density_std

        # Forward pass
        self.optimizer.zero_grad()
        prediction, tcn_repr, gcn_repr = self.model(
            density_input, flow_input,
            adj_matrix=self.adj_matrix,
            edge_index=self.edge_index
        )

        # Prediction loss (BINTS: w_prediction = 0.5)
        prediction_loss = self.prediction_criterion(prediction, density_target)

        # Temporal contrastive loss (BINTS: w_temporal = 0.25)
        # Applied on the TCN representation to enforce temporal smoothness
        try:
            temporal_loss = self.temporal_criterion(tcn_repr.permute(0, 2, 1))
            # Clamp temporal loss for stability in online setting
            temporal_loss = torch.clamp(temporal_loss, max=10.0)
        except (RuntimeError, ValueError):
            temporal_loss = torch.tensor(0.0, device=self.device)

        # Combined loss (BINTS Eq. 11)
        loss = 0.5 * prediction_loss + 0.01 * temporal_loss

        # Backward pass
        loss.backward()

        # Gradient clipping for online stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.training_losses.append(loss.item())
        self.is_trained = True

    def update(
        self,
        tracked_people: list[dict[str, Any]],
        zones: list[dict[str, Any]],
    ) -> None:
        """Feed new frame data into the predictor."""
        current_assignments = self._assign_zones(tracked_people, zones)

        # Compute signals
        raw_density = self._compute_density(current_assignments)
        smoothed_density = self._update_ema(raw_density)
        self.density_history.append(smoothed_density)

        flow = self._compute_flow(current_assignments)
        self.flow_history.append(flow)

        speeds = self._compute_zone_speeds(tracked_people, current_assignments)
        self.speed_history.append(speeds)

        self.prev_zone_assignments = current_assignments

        # Online training step every N frames
        self.frame_count += 1
        if (self.frame_count >= self.min_train_frames and
                self.frame_count % self.train_every_n == 0):
            try:
                self._train_step()
            except Exception as e:
                # Don't crash the pipeline if training fails
                print(f"[BINTS] Training step failed: {e}")

    def predict(self) -> dict[str, dict[str, Any]]:
        """
        Predict future density per zone.

        Uses neural network prediction if trained, EMA fallback otherwise.
        Both outputs are always computed for comparison.
        """
        # Warmup: return current density
        if len(self.density_history) < 3:
            result = {}
            for zone in self.zone_names:
                pred = self._empty_prediction()
                if self.density_history:
                    latest = self.density_history[-1]
                    pred["current_density"] = latest.get(zone, 0)
                    pred["predicted_density"] = latest.get(zone, 0)
                result[zone] = pred
            return result

        predictions = {}

        # ─── Neural network prediction ───
        nn_predictions = {}
        if self.is_trained and len(self.density_history) >= self.seq_len:
            try:
                self.model.eval()
                density_tensor, flow_tensor = self._prepare_tensors()
                with torch.no_grad():
                    if hasattr(self, '_density_mean'):
                        density_norm = (density_tensor - self._density_mean) / self._density_std
                    else:
                        density_norm = density_tensor
                    pred_out, _, _ = self.model(
                        density_norm, flow_tensor,
                        adj_matrix=self.adj_matrix,
                        edge_index=self.edge_index
                    )
                # Denormalize prediction
                pred_np = pred_out[0].cpu().numpy()
                if hasattr(self, '_density_mean'):
                    pred_np = pred_np * self._density_std.item() + self._density_mean.item()
                for i, zone in enumerate(self.zone_names):
                    if i < pred_np.shape[1]:
                        nn_predictions[zone] = float(pred_np[0, i])  # next step prediction
            except Exception as e:
                print(f"[BINTS] Prediction failed: {e}")

        # ─── EMA fallback prediction ───
        for zone in self.zone_names:
            density_series = [d.get(zone, 0.0) for d in self.density_history]
            current_density = density_series[-1]

            # EMA trend
            recent = density_series[-min(30, len(density_series)):]
            ema_slope = self._gaussian_weighted_trend(recent)

            # Damped EMA prediction
            phi = 0.85
            damped_horizon = sum(phi ** i for i in range(1, min(self.prediction_horizon, 10) + 1))
            ema_prediction = current_density + ema_slope * damped_horizon
            ema_prediction = max(0.0, ema_prediction)
            ema_prediction = min(ema_prediction, max(current_density * 1.5, current_density + 20))

            # Choose best prediction
            # Blend NN with EMA — NN must earn influence by reducing loss
            if zone in nn_predictions and len(self.training_losses) > 3:
                nn_pred = max(0.0, nn_predictions[zone])
                nn_pred = min(nn_pred, max(current_density * 2, current_density + 30))
                nn_weight = min(0.3, 1.0 / (self.training_losses[-1] + 1.0))
                final_prediction = nn_weight * nn_pred + (1 - nn_weight) * ema_prediction
            else:
                final_prediction = ema_prediction

            # ─── Flow analysis ───
            inflow = 0.0
            outflow = 0.0
            inflow_sources: dict[str, float] = {}
            recent_flow = list(self.flow_history)[-min(15, len(self.flow_history)):]
            for flow_snapshot in recent_flow:
                for (src, dst), count in flow_snapshot.items():
                    if dst == zone:
                        inflow += count
                        inflow_sources[src] = inflow_sources.get(src, 0.0) + count
                    if src == zone:
                        outflow += count
            net_flow = inflow - outflow

            # ─── Time to critical ───
            time_to_critical = None
            if current_density < self.critical_density and current_density > 0:
                remaining = self.critical_density - current_density
                effective_rate = ema_slope + net_flow * 0.1
                if effective_rate > 0.1:
                    raw_ttc = remaining / effective_rate
                    time_to_critical = max(5.0, min(raw_ttc, 600.0))

            # ─── Risk trend ───
            if ema_slope > 1.0 and net_flow > 1.0:
                risk_trend = "rapidly_increasing"
            elif ema_slope > 0.3 or net_flow > 0.5:
                risk_trend = "increasing"
            elif ema_slope < -1.0 and net_flow < -1.0:
                risk_trend = "rapidly_decreasing"
            elif ema_slope < -0.3 or net_flow < -0.5:
                risk_trend = "decreasing"
            else:
                risk_trend = "stable"

            # ─── Crowd pressure (Lee & Hughes 2006) ───
            crowd_pressure = 0.0
            if len(self.density_history) >= 3:
                recent_d = [d.get(zone, 0.0) for d in list(self.density_history)[-20:]]
                crowd_pressure = float(np.mean(recent_d) * np.var(recent_d))

            # ─── Primary inflow source ───
            primary_source = max(inflow_sources, key=inflow_sources.get) if inflow_sources else None

            predictions[zone] = {
                "current_density": current_density,
                "current_speed": list(self.speed_history)[-1].get(zone, 0.0) if self.speed_history else 0.0,
                "predicted_density": round(final_prediction, 2),
                "density_only_prediction": round(ema_prediction, 2),
                "nn_prediction": round(nn_predictions.get(zone, 0.0), 2),
                "predicted_inflow": round(max(0, inflow), 2),
                "predicted_outflow": round(max(0, outflow), 2),
                "density_trend_slope": round(ema_slope, 4),
                "inflow_pressure": round(inflow, 3),
                "outflow_rate": round(outflow, 3),
                "net_flow": round(net_flow, 3),
                "primary_inflow_source": primary_source,
                "spatial_pressure": 0.0,
                "crowd_pressure": round(crowd_pressure, 4),
                "time_to_critical": round(time_to_critical, 1) if time_to_critical is not None else None,
                "risk_trend": risk_trend,
                "cross_modal_alert": False,
                "bimodal_vs_density_only": round(final_prediction - ema_prediction, 2),
                "model_trained": self.is_trained,
                "training_loss": round(self.training_losses[-1], 4) if self.training_losses else None,
            }

        return predictions

    def _gaussian_weight(self, distance: int) -> float:
        return math.exp(-0.5 * (distance ** 2) / (self.gaussian_sigma ** 2))

    def _gaussian_weighted_trend(self, series: list[float]) -> float:
        n = len(series)
        if n < 2:
            return 0.0
        weights = np.array([self._gaussian_weight(n - 1 - i) for i in range(n)])
        weight_sum = weights.sum()
        if weight_sum == 0:
            return 0.0
        weights = weights / weight_sum
        x = np.arange(n, dtype=float)
        y = np.array(series, dtype=float)
        w_mean_x = np.sum(weights * x)
        w_mean_y = np.sum(weights * y)
        w_cov_xy = np.sum(weights * (x - w_mean_x) * (y - w_mean_y))
        w_var_x = np.sum(weights * (x - w_mean_x) ** 2)
        if w_var_x == 0:
            return 0.0
        return float(w_cov_xy / w_var_x)

    def get_flow_matrix(self) -> dict[str, dict[str, float]]:
        matrix = {z1: {z2: 0.0 for z2 in self.zone_names} for z1 in self.zone_names}
        recent_flow = list(self.flow_history)[-min(15, len(self.flow_history)):]
        if not recent_flow:
            return matrix
        for flow_snapshot in recent_flow:
            for (src, dst), count in flow_snapshot.items():
                if src in matrix and dst in matrix[src]:
                    matrix[src][dst] += count
        return matrix

    def get_similarity_matrix(self) -> dict[tuple[str, str], float]:
        return {}

    def _empty_prediction(self) -> dict[str, Any]:
        return {
            "current_density": 0,
            "current_speed": 0.0,
            "predicted_density": 0,
            "density_only_prediction": 0,
            "nn_prediction": 0,
            "predicted_inflow": 0,
            "predicted_outflow": 0,
            "density_trend_slope": 0,
            "inflow_pressure": 0,
            "outflow_rate": 0,
            "net_flow": 0,
            "primary_inflow_source": None,
            "spatial_pressure": 0,
            "crowd_pressure": 0,
            "time_to_critical": None,
            "risk_trend": "stable",
            "cross_modal_alert": False,
            "bimodal_vs_density_only": 0,
            "model_trained": False,
            "training_loss": None,
        }