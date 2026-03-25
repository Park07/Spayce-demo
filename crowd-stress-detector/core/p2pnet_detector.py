"""
P2PNet-based crowd detector — stable per-frame head counting.
Replaces SAHI+ByteTrack with P2PNet point-based detection.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as standard_transforms
from typing import Any

# Add P2PNet to path
P2P_ROOT = os.path.expanduser("~/CrowdCounting-P2PNet")
if P2P_ROOT not in sys.path:
    sys.path.insert(0, P2P_ROOT)

from models import build_model
import argparse


class P2PNetDetector:
    def __init__(self, weight_path=None):
        if weight_path is None:
            weight_path = os.path.join(P2P_ROOT, "weights", "SHTechA.pth")
        
        args = argparse.Namespace(
            backbone='vgg16_bn', row=2, line=2, gpu_id=0
        )
        self.device = torch.device('cpu')
        self.model = build_model(args)
        self.model.to(self.device)
        
        checkpoint = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Simple ID tracking via nearest-neighbor matching
        self._prev_points = []
        self._next_id = 0
        self._prev_ids = []
        # Trajectory history per ID
        self._trajectories = {}
    
    def detect_and_track(self, frame: np.ndarray, confidence: float = 0.5) -> list[dict[str, Any]]:
        """
        Detect head points with P2PNet and assign persistent IDs
        via nearest-neighbor matching to previous frame.
        """
        # Convert BGR (OpenCV) to RGB PIL
        img_pil = Image.fromarray(frame[:, :, ::-1]).convert('RGB')
        w, h = img_pil.size
        new_w = w // 128 * 128
        new_h = h // 128 * 128
        img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        
        # Scale factors to map back to original frame coords
        sx = w / new_w
        sy = h / new_h
        
        img_tensor = self.transform(img_pil)
        samples = torch.Tensor(img_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(samples)
        
        scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        points = outputs['pred_points'][0]
        
        # Filter by confidence
        mask = scores > confidence
        filtered_points = points[mask].cpu().numpy()
        filtered_scores = scores[mask].cpu().numpy()
        
        # Scale points back to original frame coordinates
        current_points = []
        for pt in filtered_points:
            x = int(pt[0] * sx)
            y = int(pt[1] * sy)
            current_points.append((x, y))
        
        # Assign IDs via nearest-neighbor to previous frame
        current_ids = self._match_ids(current_points)
        
        # Build tracked_people list in the format the pipeline expects
        tracked_people = []
        for i, (cx, cy) in enumerate(current_points):
            tid = current_ids[i]
            # Create a small bbox around the head point
            bbox_size = 20
            bbox = (cx - bbox_size, cy - bbox_size, cx + bbox_size, cy + bbox_size)
            
            # Update trajectory
            if tid not in self._trajectories:
                self._trajectories[tid] = []
            self._trajectories[tid].append((cx, cy))
            # Keep last 30 points
            if len(self._trajectories[tid]) > 30:
                self._trajectories[tid] = self._trajectories[tid][-30:]
            
            tracked_people.append({
                "bbox": bbox,
                "confidence": float(filtered_scores[i]),
                "centroid": (cx, cy),
                "track_id": tid,
                "trajectory": list(self._trajectories[tid]),
            })
        
        self._prev_points = current_points
        self._prev_ids = current_ids
        
        return tracked_people
    
    def _match_ids(self, current_points: list[tuple[int, int]]) -> list[int]:
        """Simple nearest-neighbor ID matching between frames."""
        if not self._prev_points or not current_points:
            # No previous frame — assign new IDs
            ids = []
            for _ in current_points:
                ids.append(self._next_id)
                self._next_id += 1
            return ids
        
        prev = np.array(self._prev_points)
        curr = np.array(current_points)
        
        # Distance matrix
        dist = np.linalg.norm(prev[:, None] - curr[None, :], axis=2)
        
        ids = [-1] * len(current_points)
        used_prev = set()
        used_curr = set()
        
        # Greedy matching: assign closest pairs first
        max_dist = 50  # pixels — max movement between frames
        flat_indices = np.argsort(dist.ravel())
        
        for flat_idx in flat_indices:
            pi = flat_idx // len(current_points)
            ci = flat_idx % len(current_points)
            if pi in used_prev or ci in used_curr:
                continue
            if dist[pi, ci] > max_dist:
                break
            ids[ci] = self._prev_ids[pi]
            used_prev.add(pi)
            used_curr.add(ci)
        
        # Assign new IDs to unmatched
        for i in range(len(ids)):
            if ids[i] == -1:
                ids[i] = self._next_id
                self._next_id += 1
        
        return ids
