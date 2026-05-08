import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from ultralytics import YOLO

# Load model directly
model = YOLO('models/yolov8n.pt')

# Read image
frame = cv2.imread('data/samples/test_vehicle.jpg')
print(f"Image shape: {frame.shape}")

# Run with very low confidence to see everything detected
results = model(frame, conf=0.15, verbose=True)

print("\n--- Raw YOLO detections ---")
for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]
            print(f"Class: {label} | Conf: {conf:.2f} | ID: {cls_id}")
    else:
        print("No boxes detected at all.")

# Save result with all detections drawn
annotated = results[0].plot()
cv2.imwrite('data/outputs/debug_result.jpg', annotated)
print("\nDebug image saved to: data/outputs/debug_result.jpg")