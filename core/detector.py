import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional
from config.settings import settings

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    0: "person"      # keep to explicitly exclude
}

VEHICLE_IDS = {2, 3, 5, 7}


@dataclass
class DetectionResult:
    vehicle_bbox: Optional[tuple] = None
    vehicle_class: Optional[str] = None
    vehicle_conf: float = 0.0
    plate_bbox: Optional[tuple] = None
    plate_crop: Optional[np.ndarray] = None
    plate_conf: float = 0.0
    track_id: Optional[int] = None


class VehicleDetector:

    def __init__(self):
        print("Loading vehicle detection model...")
        self.model = YOLO(settings.VEHICLE_MODEL_PATH)
        print("Vehicle detector ready.")

    def detect(self, frame: np.ndarray) -> list:
        results = self.model.predict(
            frame,
            conf=0.25,
            verbose=False
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])

                if cls_id not in VEHICLE_IDS:
                    continue

                conf            = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_class   = VEHICLE_CLASSES.get(cls_id, "vehicle")

                print(f"Vehicle found: {vehicle_class} | conf={conf:.2f}")

                detections.append(DetectionResult(
                    vehicle_bbox=(x1, y1, x2, y2),
                    vehicle_class=vehicle_class,
                    vehicle_conf=conf,
                    track_id=None
                ))

        return detections


class PlateDetector:

    def __init__(self):
        print("Loading plate detection model...")
        self.model = YOLO(settings.PLATE_MODEL_PATH)
        print("Plate detector ready.")

    def detect(self, frame: np.ndarray, vehicle_bbox: tuple):
        x1v, y1v, x2v, y2v = vehicle_bbox
        
        pad = 20
        x1v = max(0, x1v - pad)
        y1v = max(0, y1v - pad)
        x2v = min(frame.shape[1], x2v + pad)
        y2v = min(frame.shape[0], y2v + pad)
        
        vehicle_crop = frame[y1v:y2v, x1v:x2v]
        
        results = self.model.predict(
            vehicle_crop,
            conf=0.25,
            verbose=False
        )
        
        best_plate = None
        best_conf  = 0.0
        best_crop  = None
        
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])
                    fx1 = x1v + px1
                    fy1 = y1v + py1
                    fx2 = x1v + px2
                    fy2 = y1v + py2
                    
                    # Minimum size check — reject tiny crops
                    plate_h = fy2 - fy1
                    plate_w = fx2 - fx1
                    if plate_h < 20 or plate_w < 60:
                        print(f"Plate crop too small ({plate_w}x{plate_h}) — skipping")
                        continue
                    
                    best_plate = (fx1, fy1, fx2, fy2)
                    best_conf  = conf
                    best_crop  = frame[fy1:fy2, fx1:fx2].copy()
                    print(f"Plate found | conf={conf:.2f} | size={plate_w}x{plate_h}")
                    
        if best_plate is None:
            print("No plate detected in vehicle region.")
        
        return best_plate, best_crop, best_conf