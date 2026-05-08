import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from ultralytics import YOLO

# Load models
vehicle_model = YOLO('models/yolov8n.pt')
plate_model   = YOLO('models/plate_detector.pt')

frame = cv2.imread('data/samples/image.jpg')

# Detect vehicle
v_results = vehicle_model.predict(frame, conf=0.25, verbose=False)
for r in v_results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in {2, 3, 5, 7}:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_crop = frame[y1:y2, x1:x2]

        # Detect plate
        p_results = plate_model.predict(vehicle_crop, conf=0.25, verbose=False)
        for pr in p_results:
            for pb in pr.boxes:
                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                # Save plate crop
                cv2.imwrite('data/outputs/plate_crop.jpg', plate_crop)
                print(f"Plate crop size: {plate_crop.shape}")

                # Check HSV values
                hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
                print(f"HSV mean: {hsv.mean(axis=(0,1))}")
                print(f"HSV min:  {hsv.min(axis=(0,1))}")
                print(f"HSV max:  {hsv.max(axis=(0,1))}")

                # Test green mask with current settings
                green_mask = cv2.inRange(hsv, (36, 60, 40), (85, 255, 255))
                total = plate_crop.shape[0] * plate_crop.shape[1]
                green_pct = cv2.countNonZero(green_mask) / total
                print(f"\nGreen coverage (current settings): {green_pct:.2%}")

                # Test with wider green range
                green_mask2 = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
                green_pct2 = cv2.countNonZero(green_mask2) / total
                print(f"Green coverage (wider range):       {green_pct2:.2%}")

                # White mask
                white_mask = cv2.inRange(hsv, (0, 0, 180), (179, 60, 255))
                white_pct  = cv2.countNonZero(white_mask) / total
                print(f"White coverage:                     {white_pct:.2%}")

                print(f"\nPlate crop saved to: data/outputs/plate_crop.jpg")