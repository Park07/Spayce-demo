from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from core.config import PERSON_CLASS_ID, TRACK_MAX_DETECTIONS, TRACKER_CONFIG, YOLO_MODEL_NAME


class PersonDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME) -> None:
        self.model = YOLO(model_name)
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_name,
            confidence_threshold=0.2,
            device="cpu",
        )

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
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "centroid": centroid,
                }
            )
        return detections

    def track_people(self, frame: np.ndarray, confidence: float = 0.15) -> list[dict[str, Any]]:
        """Track people with persistent IDs using YOLO + ByteTrack.
        Uses higher resolution for better small person detection."""
        results = self.model.track(
            frame,
            verbose=False,
            conf=max(confidence, 0.15),
            iou=0.45,
            imgsz=1920,
            classes=[PERSON_CLASS_ID],
            persist=True,
            augment=True,
            tracker=TRACKER_CONFIG,
            max_det=TRACK_MAX_DETECTIONS,
        )
        tracked: list[dict[str, Any]] = []
        if not results:
            return tracked
        boxes = results[0].boxes
        if boxes is None:
            return tracked

        ids = boxes.id
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].cpu().item())
            x1, y1, x2, y2 = xyxy
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            track_id = int(ids[idx].cpu().item()) if ids is not None else -1
            tracked.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "centroid": centroid,
                    "track_id": track_id,
                }
            )
        return tracked


def draw_detections(frame: np.ndarray, tracked_people: list[dict[str, Any]]) -> np.ndarray:
    output = frame.copy()
    for person in tracked_people:
        x1, y1, x2, y2 = person["bbox"]
        track_id = person.get("track_id", -1)
        head_x = (x1 + x2) // 2
        head_y = max(0, y1 + int((y2 - y1) * 0.12))
        cv2.rectangle(output, (x1, y1), (x2, y2), (51, 153, 255), 2)
        cv2.circle(output, (head_x, head_y), 7, (0, 255, 255), 2)
        cv2.circle(output, (head_x, head_y), 2, (0, 255, 255), -1)
        cv2.putText(
            output,
            f"ID {track_id}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (51, 153, 255),
            1,
            cv2.LINE_AA,
        )
    return output
