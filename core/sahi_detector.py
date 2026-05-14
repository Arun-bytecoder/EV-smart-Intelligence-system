from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    vehicle_bbox: Optional[tuple] = None
    vehicle_class: Optional[str] = None
    vehicle_conf: float = 0.0
    plate_bbox: Optional[tuple] = None
    plate_crop: Optional[np.ndarray] = None
    plate_conf: float = 0.0
    track_id: Optional[int] = None


VEHICLE_IDS = {2, 3, 5, 7}
VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}


class SAHIVehicleDetector:

    def __init__(self, model_path: str):
        print("Loading SAHI vehicle detector...")
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.25,
            device="cpu"
        )
        print("SAHI detector ready.")

    def detect(self, frame: np.ndarray) -> list:
        result = get_sliced_prediction(
            frame,
            self.model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        detections = []
        for obj in result.object_prediction_list:
            cls_id = obj.category.id
            if cls_id not in VEHICLE_IDS:
                continue

            conf = obj.score.value
            bbox = obj.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            vehicle_class = VEHICLE_CLASSES.get(cls_id, "vehicle")

            print(f"SAHI detected: {vehicle_class} | conf={conf:.2f}")
            detections.append(DetectionResult(
                vehicle_bbox=(x1, y1, x2, y2),
                vehicle_class=vehicle_class,
                vehicle_conf=conf,
                track_id=None
            ))

        return detections