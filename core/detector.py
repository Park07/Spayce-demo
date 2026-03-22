from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from core.config import PERSON_CLASS_ID, TRACK_MAX_DETECTIONS, TRACKER_CONFIG, YOLO_MODEL_NAME
from core.tracker import CentroidTracker
from core.utils import clamp_01


class PersonDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME) -> None:
        self.model = YOLO(model_name)
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_name,
            confidence_threshold=0.2,
            device="cpu",
        )
        self.tracker = CentroidTracker(max_match_distance=55.0, max_missing_frames=18)

    def detect_people(self, frame: np.ndarray, confidence: float = 0.20) -> list[dict[str, Any]]:
        """Detect people using SAHI sliced inference for better small object detection."""
        result = get_sliced_prediction(
            frame,
            self.sahi_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0,
        )

        detections: list[dict[str, Any]] = []
        for pred in result.object_prediction_list:
            if pred.category.id != PERSON_CLASS_ID:
                continue
            bbox = pred.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            conf = pred.score.value
            if conf < confidence:
                continue
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "centroid": centroid,
                }
            )
        return self._suppress_overlaps(detections)

    def track_people(self, frame: np.ndarray, confidence: float = 0.15) -> list[dict[str, Any]]:
        """Track people with stable IDs using the detector plus centroid tracking."""
        results = self.model.track(
            frame,
            verbose=False,
            conf=max(0.15, confidence),
            iou=0.3,
            imgsz=1280,
            classes=[PERSON_CLASS_ID],
            persist=True,
            tracker=TRACKER_CONFIG,
            max_det=TRACK_MAX_DETECTIONS,
        )
        detections: list[dict[str, Any]] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().item())
            if conf < confidence:
                continue
            x1, y1, x2, y2 = xyxy
            if x2 <= x1 or y2 <= y1:
                continue
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "centroid": centroid,
                }
            )
        detections = self._suppress_overlaps(detections)
        return self.tracker.update(detections)

    def active_track_count(self) -> int:
        return self.tracker.active_count()

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
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

    def _suppress_overlaps(self, detections: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
        if len(detections) < 2:
            return detections

        ordered = sorted(detections, key=lambda item: (item.get("confidence", 0.0), self._bbox_area(item["bbox"])), reverse=True)
        kept: list[dict[str, Any]] = []
        for detection in ordered:
            if any(self._iou(detection["bbox"], existing["bbox"]) > iou_threshold for existing in kept):
                continue
            kept.append(detection)
        return kept

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(1, (x2 - x1) * (y2 - y1))


def draw_detections(frame: np.ndarray, tracked_people: list[dict[str, Any]]) -> np.ndarray:
    output = frame.copy()
    for person in tracked_people:
        x1, y1, x2, y2 = person["bbox"]
        track_id = person.get("track_id", -1)
<<<<<<< Updated upstream
        head_x = (x1 + x2) // 2
        head_y = max(0, y1 + int((y2 - y1) * 0.12))
        cv2.rectangle(output, (x1, y1), (x2, y2), (51, 153, 255), 2)
        cv2.circle(output, (head_x, head_y), 7, (0, 255, 255), 2)
        cv2.circle(output, (head_x, head_y), 2, (0, 255, 255), -1)
=======

        # Compute bounding region for the head/face (top 20% of the bounding box)
        head_x = (x1 + x2) // 2
        head_y = max(0, y1 + int((y2 - y1) * 0.12))
        head_h_region = int((y2 - y1) * 0.25)
        head_w_region = int((x2 - x1) * 0.5)
        
        hx1 = max(0, head_x - head_w_region)
        hy1 = max(0, y1)
        hx2 = min(output.shape[1], head_x + head_w_region)
        hy2 = min(output.shape[0], y1 + head_h_region)

        # Apply Privacy Face Blurring (Execution Layer Step 4)
        if anonymize and (hx2 > hx1) and (hy2 > hy1):
            roi = output[hy1:hy2, hx1:hx2]
            # Intense blur to completely anonymize the face
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            output[hy1:hy2, hx1:hx2] = blurred_roi

        risk_score = clamp_01(float(person.get("risk_score", person.get("severity_score", 0.0))))
        if risk_score >= 0.7:
            box_color = (0, 0, 255)
        elif risk_score >= 0.35:
            box_color = (0, 215, 255)
        else:
            box_color = (0, 200, 0)

        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, thickness=-1)
        cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(output, (x1, y1), (min(x2, x1 + 160), min(y2, y1 + 22)), box_color, -1)
>>>>>>> Stashed changes
        cv2.putText(
            output,
            f"ID {track_id} | {risk_score:.2f}",
            (x1 + 4, min(y1 + 16, output.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f"{'HIGH' if risk_score >= 0.7 else 'MED' if risk_score >= 0.35 else 'LOW'}",
            (x1 + 4, min(y1 + 36, output.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )
    return output
