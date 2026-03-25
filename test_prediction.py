"""
Test the prediction layer honestly.
For each frame, record what the predictor says will happen in N frames.
Then check what actually happened N frames later.
"""
import sys
import math
import numpy as np
from collections import deque
from core.prediction import BiModalPredictor

# P2PNet actual counts from Tokyo video (55 frames at 5fps)
actual_counts = [
    74, 76, 75, 73, 80, 77, 73, 74, 64, 62,
    62, 66, 47, 54, 50, 51, 64, 59, 48, 51,
    74, 72, 47, 61, 91, 66, 50, 65, 95, 145,
    89, 85, 92, 78, 88, 54, 76, 60, 65, 78,
    73, 77, 67, 93, 89, 121, 107, 70, 64, 50,
    70, 41, 39, 37, 35,
]

# Simulate the predictor with single zone "Camera View"
predictor = BiModalPredictor(
    zone_names=["Camera View"],
    zone_types={"Camera View": "bottleneck"},
    window_size=90,
    prediction_horizon=5,  # predict 5 frames ahead (1 second at 5fps)
    critical_density_per_zone=80,
)

# Fake zone covering entire frame
zone = {"name": "Camera View", "type": "bottleneck", "rect": (0, 0, 1920, 1080)}

predictions_log = []

for i, count in enumerate(actual_counts):
    # Create fake tracked people with centroids spread across frame
    tracked_people = []
    for p in range(int(count)):
        x = (p * 47) % 1920  # spread across frame
        y = (p * 31) % 1080
        tracked_people.append({
            "track_id": p,
            "centroid": (x, y),
            "bbox": (x-10, y-10, x+10, y+10),
            "trajectory": [(x, y)],
        })
    
    # Feed to predictor
    predictor.update(tracked_people, [zone])
    
    # Get prediction
    pred = predictor.predict()
    zp = pred.get("Camera View", {})
    
    predictions_log.append({
        "frame": i,
        "time": round(i * 0.2, 1),
        "actual": count,
        "predicted": zp.get("predicted_density", 0),
        "trend": zp.get("risk_trend", "unknown"),
        "ttc": zp.get("time_to_critical"),
        "slope": zp.get("density_trend_slope", 0),
    })

# Now evaluate: compare prediction at frame N with actual at frame N+5
LOOKAHEAD = 5
print("=" * 80)
print(f"PREDICTION VALIDATION — {LOOKAHEAD}-frame lookahead (1 second)")
print("=" * 80)
print(f"{'Frame':>5} {'Time':>5} {'Actual':>7} {'Predicted':>9} {'Actual+5':>9} {'Error':>7} {'Trend':>20} {'TTC':>8}")
print("-" * 80)

errors = []
direction_correct = 0
direction_total = 0

for i in range(len(predictions_log) - LOOKAHEAD):
    p = predictions_log[i]
    future_actual = actual_counts[i + LOOKAHEAD]
    predicted = p["predicted"]
    error = abs(predicted - future_actual)
    errors.append(error)
    
    # Check if prediction got the DIRECTION right
    current = p["actual"]
    pred_direction = "up" if predicted > current else "down" if predicted < current else "flat"
    actual_direction = "up" if future_actual > current else "down" if future_actual < current else "flat"
    direction_match = pred_direction == actual_direction
    if pred_direction != "flat":
        direction_total += 1
        if direction_match:
            direction_correct += 1
    
    ttc_str = f"{p['ttc']:.0f}s" if p['ttc'] is not None else "—"
    match_symbol = "✓" if direction_match else "✗"
    
    print(f"{p['frame']:>5} {p['time']:>5} {p['actual']:>7} {predicted:>9.1f} {future_actual:>9} {error:>7.1f} {p['trend']:>20} {ttc_str:>8} {match_symbol}")

print("=" * 80)
print(f"\nRESULTS:")
print(f"  MAE (Mean Absolute Error):     {np.mean(errors):.1f} people")
print(f"  RMSE:                          {np.sqrt(np.mean(np.array(errors)**2)):.1f} people")
print(f"  Max Error:                     {max(errors):.1f} people")
print(f"  Min Error:                     {min(errors):.1f} people")
print(f"  Direction Accuracy:            {direction_correct}/{direction_total} ({100*direction_correct/max(direction_total,1):.0f}%)")
print(f"  Frames with error < 10:        {sum(1 for e in errors if e < 10)}/{len(errors)} ({100*sum(1 for e in errors if e < 10)/len(errors):.0f}%)")
print(f"  Frames with error < 20:        {sum(1 for e in errors if e < 20)}/{len(errors)} ({100*sum(1 for e in errors if e < 20)/len(errors):.0f}%)")

# Time-to-critical validation
print(f"\n{'='*80}")
print("TIME-TO-CRITICAL VALIDATION")
print(f"{'='*80}")
print(f"Capacity threshold: 80 people")
print(f"Frames where actual exceeded 80: {[i for i, c in enumerate(actual_counts) if c > 80]}")
print(f"\nTTC predictions before threshold crossings:")
for i in range(len(predictions_log)):
    p = predictions_log[i]
    if p["ttc"] is not None and p["actual"] < 80:
        # Find when actual actually exceeded 80
        exceeded_frame = None
        for j in range(i+1, len(actual_counts)):
            if actual_counts[j] > 80:
                exceeded_frame = j
                break
        if exceeded_frame:
            actual_time_to_exceed = (exceeded_frame - i) * 0.2  # seconds
            print(f"  Frame {i} (t={p['time']}s): Predicted critical in {p['ttc']:.1f}s | Actually exceeded at frame {exceeded_frame} ({actual_time_to_exceed:.1f}s later) | Error: {abs(p['ttc'] - actual_time_to_exceed):.1f}s")
