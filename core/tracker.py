from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Any


class CentroidTracker:
    """A lightweight centroid tracker for demo-friendly ID persistence."""

    def __init__(self, max_match_distance: float = 35.0, max_missing_frames: int = 20) -> None:
        self.max_match_distance = max_match_distance
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.active_tracks: dict[int, dict[str, Any]] = {}
        self.missing_counts: dict[int, int] = defaultdict(int)
        self.history: dict[int, list[tuple[int, int]]] = defaultdict(list)

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        matched_track_ids: set[int] = set()

        for det in sorted(detections, key=lambda item: item.get("confidence", 0.0), reverse=True):
            centroid = det["centroid"]
            bbox = det.get("bbox")
            best_track = None
            best_score = float("inf")

            for track_id, track in self.active_tracks.items():
                if track_id in matched_track_ids:
                    continue
                prev_center = track["centroid"]
                dist = hypot(centroid[0] - prev_center[0], centroid[1] - prev_center[1])
                if dist > self.max_match_distance:
                    continue
                score = dist
                if bbox is not None and track.get("bbox") is not None:
                    score *= 1.0 - min(0.5, self._bbox_iou(bbox, track["bbox"]))
                if score < best_score:
                    best_score = score
                    best_track = track_id

            if best_track is None:
                best_track = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[best_track] = {
                    "centroid": centroid,
                    "bbox": bbox,
                    "confidence": det.get("confidence", 0.0),
                    "hits": 0,
                }

            track = self.active_tracks[best_track]
            track["centroid"] = centroid
            track["bbox"] = bbox
            track["confidence"] = det.get("confidence", 0.0)
            track["hits"] = int(track.get("hits", 0)) + 1
            self.missing_counts[best_track] = 0
            self.history[best_track].append(centroid)
            if len(self.history[best_track]) > 40:
                self.history[best_track] = self.history[best_track][-40:]

            det["track_id"] = best_track
            det["trajectory"] = list(self.history[best_track])
            updated.append(det)
            matched_track_ids.add(best_track)

        for track_id in list(self.active_tracks.keys()):
            if track_id in matched_track_ids:
                continue
            self.missing_counts[track_id] += 1
            if self.missing_counts[track_id] > self.max_missing_frames:
                self.active_tracks.pop(track_id, None)
                self.missing_counts.pop(track_id, None)
                self.history.pop(track_id, None)

        return updated

    def active_count(self) -> int:
        return len(self.active_tracks)

    @staticmethod
    def _bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
        area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
        area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union
