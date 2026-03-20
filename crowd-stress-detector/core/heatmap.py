from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from core.config import HEATMAP_BLUR_KERNEL


def build_heatmap_overlay(
    frame_shape: tuple[int, int, int], centroids: Iterable[tuple[int, int]], alpha: float = 0.45
) -> np.ndarray:
    h, w = frame_shape[:2]
    density = np.zeros((h, w), dtype=np.float32)
    for x, y in centroids:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(density, (x, y), 24, 1.0, thickness=-1)

    density = cv2.GaussianBlur(density, HEATMAP_BLUR_KERNEL, 0)
    if density.max() > 0:
        density = density / density.max()
    heatmap_u8 = (density * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(colored, alpha, np.zeros_like(colored), 1 - alpha, 0)


def overlay_heatmap(frame: np.ndarray, heatmap_layer: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    return cv2.addWeighted(frame, 1.0, heatmap_layer, alpha, 0)
