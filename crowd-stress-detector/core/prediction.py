"""
Bi-Modal Crowd Prediction Layer
================================
Inspired by BINTS: Bi-Modal Learning for Networked Time Series (KDD 2025)
Paper: https://doi.org/10.1145/3711896.3736856
Authors: Nam et al., KAIST

Core concepts adapted from the paper:
- Section 4.3: Separate encoders for node-oriented (density) and edge-oriented (flow) time series
- Section 4.4: Spatial contrastive learning via cosine similarity aligned with adjacency structure
- Section 4.5: Temporal contrastive learning via Gaussian-weighted similarity across time steps
- Section 4.6: Prediction using combined representation with loss weights 0.5/0.25/0.25
- Table 4: Ablation showing both spatial and temporal components are critical (~17% each)

Adaptation for real-time video pipeline:
- Node-oriented time series = person count per zone per frame (from YOLOv8 + ByteTrack)
- Edge-oriented time series = tracked ID flow between zones per frame
- Adjacency matrix = zone connectivity defined in config (k-hop with exponential decay)
- Prediction uses lightweight math instead of TCN/GCN (no historical training corpus)

Research-backed thresholds from:
- Itaewon stampede analysis (ScienceDirect 2025): critical density 6.875-6.971 ped/m²
- Keith Still PhD (crowd safety pioneer): time-to-critical as key metric
- PLOS ONE 2021 entropy model: critical stationary density 4.7 ped/m², moving 4.0 ped/m²
- Crowd pressure = density × variance (from "Prediction of Human Crowd Pressures", 2006)
"""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from typing import Any

from core.config import ZONE_ADJACENCY, PREDICTION_WEIGHTS


class BiModalPredictor:
    """
    BINTS-inspired bi-modal predictor combining zone density (node features)
    and inter-zone flow (edge features) for crowd density prediction.

    Maps to BINTS architecture:
    - density_history -> node-oriented time series V_t^ri (Section 4.3, Eq. 1)
    - flow_history -> edge-oriented time series F_t^ri (Section 4.3, Eq. 2)
    - _compute_cross_modal_similarity -> 3D similarity tensor S (Section 4.4, Eq. 3)
    - _gaussian_weight -> temporal contrastive weighting (Section 4.5, Eq. 8)
    - _adjacency_weight -> k-hop spatial structure (Section 4.1)
    - predict() -> target node attribute prediction (Section 4.6, Eq. 10)
    """

    def __init__(
        self,
        zone_names: list[str],
        zone_types: dict[str, str] | None = None,
        window_size: int = 90,
        prediction_horizon: int = 30,
        gaussian_sigma: float = 10.0,
        critical_density_per_zone: float = 20.0,
    ):
        """
        Args:
            zone_names: list of zone identifiers matching config
            zone_types: dict mapping zone name to type (exit/entry/bottleneck)
            window_size: rolling history length in frames
            prediction_horizon: frames ahead to predict
            gaussian_sigma: sigma for temporal Gaussian weighting (Section 4.5)
            critical_density_per_zone: person count threshold per zone for time-to-critical
        """
        self.zone_names = zone_names
        self.zone_types = zone_types or {}
        self.num_zones = len(zone_names)
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.gaussian_sigma = gaussian_sigma
        self.critical_density = critical_density_per_zone

        # Node-oriented time series: density per zone over time
        # Maps to V_t^ri in BINTS (Section 4.3)
        self.density_history: deque[dict[str, float]] = deque(maxlen=window_size)

        # Edge-oriented time series: flow between zones over time
        # Maps to F_t^ri in BINTS (Section 4.3)
        self.flow_history: deque[dict[tuple[str, str], float]] = deque(maxlen=window_size)

        # Cross-modal similarity history: cosine sim between density and flow vectors
        # Maps to S tensor in BINTS (Section 4.4, Eq. 3)
        self.similarity_history: deque[dict[tuple[str, str], float]] = deque(maxlen=window_size)

        # Track which IDs were in which zone last frame (for computing flow)
        self.prev_zone_assignments: dict[int, str] = {}

        # Per-zone speed history for crowd pressure computation
        self.speed_history: deque[dict[str, float]] = deque(maxlen=window_size)

        # Build adjacency matrix from config
        # Maps to k-hop adjacency G^(k) in BINTS (Section 4.1)
        self.adjacency = self._build_adjacency()

    def _build_adjacency(self) -> dict[tuple[str, str], float]:
        """
        Build full adjacency matrix including self-loops.
        From BINTS Section 4.1: "all diagonal elements are set to g_ii = 1;
        the inclusion of self-loops allows the model to capture the self-influence
        of its features and interactions."
        """
        adj = {}
        # Self-loops (Section 4.1: g_ii = 1)
        for z in self.zone_names:
            adj[(z, z)] = 1.0
        # From config adjacency (with k-hop exponential decay weights)
        for (z1, z2), weight in ZONE_ADJACENCY.items():
            if z1 in self.zone_names and z2 in self.zone_names:
                adj[(z1, z2)] = weight
                adj[(z2, z1)] = weight  # undirected for spatial similarity
        return adj

    def _assign_zones(
        self, tracked_people: list[dict[str, Any]], zones: list[dict[str, Any]]
    ) -> dict[int, str]:
        """Assign each tracked person to a zone based on centroid location."""
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
        """
        Count people per zone = node-oriented time series.
        Maps to V_t^ri in BINTS.
        """
        density = {z: 0.0 for z in self.zone_names}
        for zone in assignments.values():
            if zone in density:
                density[zone] += 1.0
        return density

    def _compute_flow(self, current_assignments: dict[int, str]) -> dict[tuple[str, str], float]:
        """
        Track ID movement between zones = edge-oriented time series.
        Maps to F_t^ri (origin-destination flows) in BINTS.
        """
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
        """Compute average movement speed per zone for crowd pressure metric."""
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

    def _compute_cross_modal_similarity(
        self, density: dict[str, float], flow: dict[tuple[str, str], float]
    ) -> dict[tuple[str, str], float]:
        """
        Compute cosine similarity between density vector and flow vector per zone pair.

        Maps to BINTS Section 4.4, Equation 3:
        s_t^(ri, rj) = (h_TCN^ri · h_GCN^rj) / (||h_TCN^ri|| · ||h_GCN^rj||)

        Simplified: for each zone pair (i, j), compute how similar zone i's
        density pattern is to zone j's flow pattern. High similarity between
        connected zones = expected. High similarity between disconnected zones
        = anomaly. Diverging similarity between connected zones = danger signal.
        """
        similarities = {}

        # Build density vector per zone (normalized)
        density_vals = np.array([density.get(z, 0.0) for z in self.zone_names])
        density_norm = np.linalg.norm(density_vals)
        if density_norm > 0:
            density_vals = density_vals / density_norm

        # Build flow vector per zone (total inflow to each zone)
        flow_vals = np.zeros(self.num_zones)
        for (src, dst), count in flow.items():
            if dst in self.zone_names:
                dst_idx = self.zone_names.index(dst)
                flow_vals[dst_idx] += count
        flow_norm = np.linalg.norm(flow_vals)
        if flow_norm > 0:
            flow_vals = flow_vals / flow_norm

        # Cosine similarity between each zone pair's density and flow
        for i, zi in enumerate(self.zone_names):
            for j, zj in enumerate(self.zone_names):
                if density_norm > 0 and flow_norm > 0:
                    # Element-wise contribution to similarity
                    sim = float(density_vals[i] * flow_vals[j])
                else:
                    sim = 0.0
                similarities[(zi, zj)] = sim

        return similarities

    def _gaussian_weight(self, distance: int) -> float:
        """
        Gaussian weighting for temporal smoothing.

        From BINTS Section 4.5, Equation 8:
        w(t, t') = G(t - t') / sum(G(t - k))
        where G(x) = (1/σ√2π) exp(-x²/2σ²)

        Adjacent timesteps get higher weight than distant ones.
        The ablation (Table 4) shows removing this costs 17.6% accuracy.
        """
        return math.exp(-0.5 * (distance ** 2) / (self.gaussian_sigma ** 2))

    def _gaussian_weighted_trend(self, series: list[float]) -> float:
        """
        Compute trend slope using Gaussian-weighted regression.
        Recent observations are weighted exponentially more than older ones.
        """
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

        # Weighted linear regression
        w_mean_x = np.sum(weights * x)
        w_mean_y = np.sum(weights * y)
        w_cov_xy = np.sum(weights * (x - w_mean_x) * (y - w_mean_y))
        w_var_x = np.sum(weights * (x - w_mean_x) ** 2)

        if w_var_x == 0:
            return 0.0
        return float(w_cov_xy / w_var_x)

    def _compute_crowd_pressure(self, zone: str) -> float:
        """
        Crowd pressure = density × variance of density.

        From "Prediction of Human Crowd Pressures" (Lee & Hughes, 2006):
        "crowd pressure, a product between the density and the variance of
        the densities, gave better information about critical areas and timings"

        This is better than raw density because a uniform crowd at 4 ped/m²
        is less dangerous than an uneven crowd averaging 4 ped/m² with spikes.
        """
        if len(self.density_history) < 3:
            return 0.0

        recent_densities = [d.get(zone, 0.0) for d in list(self.density_history)[-20:]]
        mean_density = np.mean(recent_densities)
        density_variance = np.var(recent_densities)

        return float(mean_density * density_variance)

    def update(
        self,
        tracked_people: list[dict[str, Any]],
        zones: list[dict[str, Any]],
    ) -> None:
        """
        Feed new frame data into the predictor.
        Called every frame from video_processor.py.
        """
        # Zone assignment
        current_assignments = self._assign_zones(tracked_people, zones)

        # Node features: density per zone (BINTS V_t^ri)
        density = self._compute_density(current_assignments)
        self.density_history.append(density)

        # Edge features: flow between zones (BINTS F_t^ri)
        flow = self._compute_flow(current_assignments)
        self.flow_history.append(flow)

        # Cross-modal similarity (BINTS Section 4.4, Eq. 3)
        similarity = self._compute_cross_modal_similarity(density, flow)
        self.similarity_history.append(similarity)

        # Per-zone speeds for crowd pressure
        speeds = self._compute_zone_speeds(tracked_people, current_assignments)
        self.speed_history.append(speeds)

        # Update zone assignments for next frame's flow computation
        self.prev_zone_assignments = current_assignments

    def predict(self) -> dict[str, dict[str, Any]]:
        """
        Predict future density per zone using bi-modal approach.

        Architecture maps to BINTS Section 4.6:
        - prediction_signal: direct trend extrapolation (w_prediction = 0.5)
        - spatial_signal: adjacency-weighted cross-modal similarity (w_spatial = 0.25)
        - temporal_signal: Gaussian-smoothed trend (w_temporal = 0.25)

        Returns per-zone prediction dict with:
        - current_density, predicted_density, density_only_prediction
        - crowd_pressure (density × variance, from Lee & Hughes 2006)
        - inflow_pressure, outflow_rate, net_flow
        - time_to_critical (Keith Still's key metric)
        - risk_trend classification
        - primary_inflow_source
        - cross_modal_alert (when density-flow similarity diverges from adjacency)
        """
        if len(self.density_history) < 5:
            return {zone: self._empty_prediction() for zone in self.zone_names}

        w_pred = PREDICTION_WEIGHTS["prediction"]   # 0.5
        w_spatial = PREDICTION_WEIGHTS["spatial"]    # 0.25
        w_temporal = PREDICTION_WEIGHTS["temporal"]  # 0.25

        predictions = {}

        for zone in self.zone_names:
            # === NODE SIGNAL: density over time (BINTS V_t^ri) ===
            density_series = [d.get(zone, 0.0) for d in self.density_history]
            current_density = density_series[-1]

            # Gaussian-weighted trend (BINTS temporal contrastive, Section 4.5)
            recent_window = density_series[-min(30, len(density_series)):]
            density_slope = self._gaussian_weighted_trend(recent_window)

            # Direct prediction signal: extrapolate Gaussian-weighted trend
            density_prediction = current_density + density_slope * self.prediction_horizon

            # === EDGE SIGNAL: flow into/out of zone (BINTS F_t^ri) ===
            inflow = 0.0
            outflow = 0.0
            inflow_sources: dict[str, float] = {}

            recent_flow = list(self.flow_history)[-min(15, len(self.flow_history)):]
            # Gaussian-weighted flow averaging
            flow_weights = [self._gaussian_weight(len(recent_flow) - 1 - i) for i in range(len(recent_flow))]
            flow_weight_sum = sum(flow_weights) if flow_weights else 1.0

            for idx, flow_snapshot in enumerate(recent_flow):
                w = flow_weights[idx] / flow_weight_sum if flow_weight_sum > 0 else 0.0
                for (src, dst), count in flow_snapshot.items():
                    if dst == zone:
                        inflow += count * w
                        inflow_sources[src] = inflow_sources.get(src, 0.0) + count * w
                    if src == zone:
                        outflow += count * w

            net_flow = inflow - outflow

            # === SPATIAL SIGNAL: adjacency-weighted cross-modal similarity ===
            # (BINTS Section 4.4, spatial contrastive learning)
            # For each connected zone, check if the cross-modal similarity
            # matches the adjacency expectation. Deviation = anomaly signal.
            spatial_pressure = 0.0
            recent_sim = list(self.similarity_history)[-min(10, len(self.similarity_history)):]

            for other_zone in self.zone_names:
                if other_zone == zone:
                    continue
                adj_weight = self.adjacency.get((other_zone, zone), 0.0)
                if adj_weight <= 0:
                    continue

                # Average recent cross-modal similarity for this pair
                pair_sims = []
                for sim_snapshot in recent_sim:
                    pair_sims.append(sim_snapshot.get((other_zone, zone), 0.0))

                if pair_sims:
                    avg_sim = float(np.mean(pair_sims))
                    # Spatial pressure: adjacency weight × similarity
                    # High adj_weight × high similarity = strong inflow signal
                    spatial_pressure += adj_weight * avg_sim

                    # Check for flow from connected zones
                    other_density_series = [d.get(other_zone, 0.0) for d in self.density_history]
                    if len(other_density_series) >= 5:
                        other_slope = self._gaussian_weighted_trend(
                            other_density_series[-min(30, len(other_density_series)):]
                        )
                        # If connected zone is losing people (negative slope),
                        # they might be flowing into this zone
                        if other_slope < 0:
                            spatial_pressure += adj_weight * abs(other_slope)

            # === BI-MODAL COMBINATION (BINTS Section 4.6, Eq. 11) ===
            # L_total = w_prediction * L_pred + w_spatial * L_spatial + w_temporal * L_temporal
            # Adapted: predicted_density = w_pred * trend_prediction
            #                            + w_spatial * spatial_adjustment
            #                            + w_temporal * temporal_smoothed_prediction

            # Prediction signal: raw trend extrapolation
            prediction_signal = density_prediction

            # Spatial signal: flow + adjacency-weighted pressure
            spatial_adjustment = current_density + (net_flow + spatial_pressure) * self.prediction_horizon

            # Temporal signal: Gaussian-smoothed current state projected forward
            # Uses the smoothed slope which already incorporates Gaussian weighting
            temporal_signal = current_density + density_slope * self.prediction_horizon * 0.8

            # Weighted combination (Table 5: optimal at 0.5/0.25/0.25)
            bi_modal_prediction = (
                w_pred * prediction_signal
                + w_spatial * spatial_adjustment
                + w_temporal * temporal_signal
            )
            bi_modal_prediction = max(0.0, bi_modal_prediction)

            # === CROWD PRESSURE (Lee & Hughes 2006) ===
            crowd_pressure = self._compute_crowd_pressure(zone)

            # === TIME TO CRITICAL (Keith Still) ===
            time_to_critical = None
            if current_density < self.critical_density:
                remaining = self.critical_density - current_density
                # Effective rate combines density trend + flow-adjusted rate
                # weighted by the same 50/25/25 scheme
                effective_rate = (
                    w_pred * density_slope
                    + w_spatial * (net_flow + spatial_pressure)
                    + w_temporal * (density_slope * 0.8)
                )
                if effective_rate > 0.01:  # minimum threshold to avoid division by tiny numbers
                    time_to_critical = remaining / effective_rate

            # === CROSS-MODAL ALERT ===
            # When density and flow patterns diverge for connected zones,
            # it signals an unexpected crowd dynamic
            cross_modal_alert = False
            if len(self.similarity_history) >= 10:
                recent_sims = [
                    sum(s.get((z, zone), 0.0) for z in self.zone_names if z != zone)
                    for s in list(self.similarity_history)[-10:]
                ]
                if len(recent_sims) >= 5:
                    sim_slope = self._gaussian_weighted_trend(recent_sims)
                    # Rapidly changing cross-modal similarity = unstable crowd dynamics
                    if abs(sim_slope) > 0.05:
                        cross_modal_alert = True

            # === RISK TREND ===
            if density_slope > 0.5 or net_flow > 2.0 or spatial_pressure > 1.0:
                risk_trend = "rapidly_increasing"
            elif density_slope > 0.1 or net_flow > 0.5 or spatial_pressure > 0.3:
                risk_trend = "increasing"
            elif density_slope < -0.5 or net_flow < -2.0:
                risk_trend = "rapidly_decreasing"
            elif density_slope < -0.1 or net_flow < -0.5:
                risk_trend = "decreasing"
            else:
                risk_trend = "stable"

            # === PRIMARY INFLOW SOURCE ===
            primary_source = None
            if inflow_sources:
                primary_source = max(inflow_sources, key=inflow_sources.get)

            # === PREDICTED FLOW (BINTS predicts both density AND flow) ===
            # Definition 4.1: target attributes X = {V (density), F (flow)}
            predicted_inflow = inflow * (1 + density_slope * 0.1) if density_slope > 0 else inflow * 0.9
            predicted_outflow = outflow * (1 - density_slope * 0.1) if density_slope > 0 else outflow * 1.1

            predictions[zone] = {
                # Current state
                "current_density": current_density,
                "current_speed": list(self.speed_history)[-1].get(zone, 0.0) if self.speed_history else 0.0,

                # Predictions
                "predicted_density": round(bi_modal_prediction, 2),
                "density_only_prediction": round(density_prediction, 2),
                "predicted_inflow": round(max(0, predicted_inflow), 2),
                "predicted_outflow": round(max(0, predicted_outflow), 2),

                # Trend analysis (Gaussian-weighted, BINTS Section 4.5)
                "density_trend_slope": round(density_slope, 4),

                # Flow analysis (edge-oriented, BINTS Section 4.3)
                "inflow_pressure": round(inflow, 3),
                "outflow_rate": round(outflow, 3),
                "net_flow": round(net_flow, 3),
                "primary_inflow_source": primary_source,

                # Spatial analysis (adjacency-weighted, BINTS Section 4.4)
                "spatial_pressure": round(spatial_pressure, 3),

                # Crowd safety metrics (research-backed)
                "crowd_pressure": round(crowd_pressure, 4),
                "time_to_critical": round(time_to_critical, 1) if time_to_critical is not None else None,

                # Alerts
                "risk_trend": risk_trend,
                "cross_modal_alert": cross_modal_alert,

                # Meta: comparison showing bi-modal advantage
                "bimodal_vs_density_only": round(bi_modal_prediction - density_prediction, 2),
            }

        return predictions

    def get_flow_matrix(self) -> dict[str, dict[str, float]]:
        """
        Get current Gaussian-weighted flow matrix between all zones.
        Maps to the OD matrix used in BINTS data_loader.py.
        """
        matrix = {z1: {z2: 0.0 for z2 in self.zone_names} for z1 in self.zone_names}
        recent_flow = list(self.flow_history)[-min(15, len(self.flow_history)):]
        if not recent_flow:
            return matrix

        weights = [self._gaussian_weight(len(recent_flow) - 1 - i) for i in range(len(recent_flow))]
        weight_sum = sum(weights) if weights else 1.0

        for idx, flow_snapshot in enumerate(recent_flow):
            w = weights[idx] / weight_sum if weight_sum > 0 else 0.0
            for (src, dst), count in flow_snapshot.items():
                if src in matrix and dst in matrix[src]:
                    matrix[src][dst] += count * w
        return matrix

    def get_similarity_matrix(self) -> dict[tuple[str, str], float]:
        """
        Get current averaged cross-modal similarity matrix.
        Maps to the 3D similarity tensor S in BINTS (Section 4.4).
        """
        if not self.similarity_history:
            return {}

        recent = list(self.similarity_history)[-min(10, len(self.similarity_history)):]
        averaged = {}
        for zi in self.zone_names:
            for zj in self.zone_names:
                vals = [s.get((zi, zj), 0.0) for s in recent]
                averaged[(zi, zj)] = float(np.mean(vals)) if vals else 0.0
        return averaged

    def _empty_prediction(self) -> dict[str, Any]:
        """Return empty prediction when not enough data accumulated."""
        return {
            "current_density": 0,
            "current_speed": 0.0,
            "predicted_density": 0,
            "density_only_prediction": 0,
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
        }