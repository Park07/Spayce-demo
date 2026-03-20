from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Any


class CentroidTracker:
    """A lightweight centroid tracker for demo-friendly ID persistence."""

    def __init__(self, max_match_distance: float = 65.0, max_missing_frames: int = 12) -> None:
        self.max_match_distance = max_match_distance
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.active_tracks: dict[int, tuple[int, int]] = {}
        self.missing_counts: dict[int, int] = defaultdict(int)
        self.history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        unmatched_track_ids = set(self.active_tracks.keys())

        for det in detections:
            centroid = det["centroid"]
            best_track = None
            best_dist = float("inf")
            for track_id, prev_center in self.active_tracks.items():
                dist = hypot(centroid[0] - prev_center[0], centroid[1] - prev_center[1])
                if dist < best_dist and dist <= self.max_match_distance:
                    best_dist = dist
                    best_track = track_id

            if best_track is None:
                best_track = self.next_track_id
                self.next_track_id += 1

            self.active_tracks[best_track] = centroid
            self.missing_counts[best_track] = 0
            self.history[best_track].append(centroid)
            if len(self.history[best_track]) > 40:
                self.history[best_track] = self.history[best_track][-40:]

            det["track_id"] = best_track
            det["trajectory"] = self.history[best_track]
            updated.append(det)
            unmatched_track_ids.discard(best_track)

        for track_id in list(unmatched_track_ids):
            self.missing_counts[track_id] += 1
            if self.missing_counts[track_id] > self.max_missing_frames:
                self.active_tracks.pop(track_id, None)
                self.missing_counts.pop(track_id, None)
                self.history.pop(track_id, None)

        return updated
