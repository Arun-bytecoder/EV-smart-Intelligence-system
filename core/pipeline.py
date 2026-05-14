import cv2
import numpy as np
from typing import Optional

from core.preprocessor import Preprocessor, PlateEnhancer
from core.detector import VehicleDetector, PlateDetector, DetectionResult
from core.color_classifier import ColorClassifier, PlateColor
from core.ocr_engine import OCREngine
from core.confidence_engine import ConfidenceEngine, ConfidenceResult, EVDecision
from api.vahan_client import VahanClient
from utils.annotator import Annotator
from utils.logger import DetectionLogger
from config.settings import settings


class EVDetectionPipeline:

    def __init__(self):
        print("Initializing EV Smart Detection Pipeline...")
        self.preprocessor      = Preprocessor()
        self.vehicle_detector  = VehicleDetector()
        self.plate_detector    = PlateDetector()
        self.color_classifier  = ColorClassifier()
        self.ocr_engine        = OCREngine()
        self.confidence_engine = ConfidenceEngine()
        self.vahan_client      = VahanClient()
        self.annotator         = Annotator()
        self.detection_logger  = DetectionLogger()
        self.enhancer          = PlateEnhancer()

        self._processed_tracks = {}
        self._frame_count      = 0
        print("Pipeline ready.")

    def process_frame(self, frame: np.ndarray):
        self._frame_count += 1
        results = []

        # Stage 0: preprocess
        enhanced = self.preprocessor.enhance_frame(frame)

        # Stage 1: vehicle detection
        detections = self.vehicle_detector.detect(enhanced)

        if not detections:
            annotated = self.annotator.draw_no_detection(frame)
            return annotated, results

        for det in detections:
            # Use cached result if vehicle already processed
            if det.track_id and det.track_id in self._processed_tracks:
                cached = self._processed_tracks[det.track_id]
                frame  = self.annotator.draw_result(frame, det, cached)
                results.append(cached)
                continue

            # Stage 2: plate detection
            plate_bbox, plate_crop, plate_conf = self.plate_detector.detect(
                enhanced, det.vehicle_bbox
            )

            if plate_crop is None:
                x1, y1, x2, y2 = det.vehicle_bbox
                vh = y2 - y1
                vw = x2 - x1
                
                # Try multiple crop regions — different vehicles have plates at different positions
                crop_regions = []
                
                if det.vehicle_class == "motorcycle":
                    crop_regions = [
                        # Primary: 35-55% height, center 50% width
                        (y1 + int(vh * 0.35), y1 + int(vh * 0.55),
                         x1 + int(vw * 0.25), x2 - int(vw * 0.25)),
                        # Secondary: 30-60% height, wider
                        (y1 + int(vh * 0.30), y1 + int(vh * 0.60),
                         x1 + int(vw * 0.15), x2 - int(vw * 0.15)),
                        # Tertiary: 20-50% height
                        (y1 + int(vh * 0.20), y1 + int(vh * 0.50),
                         x1 + int(vw * 0.20), x2 - int(vw * 0.20)),
                    ]
                else:
                    # Cars/buses/trucks: plate is at bottom (70-100% height)
                    crop_regions = [
                        (y1 + int(vh * 0.72), y2,
                         x1 + int(vw * 0.20), x2 - int(vw * 0.20)),
                        (y1 + int(vh * 0.65), y2, x1, x2),
                    ]
                    
                extracted = False
                for (fy1, fy2, fx1, fx2) in crop_regions:
                    fallback_crop = enhanced[fy1:fy2, fx1:fx2]
                    if fallback_crop.size == 0:
                        continue
                    print(f"Trying fallback crop: {fallback_crop.shape} for {det.vehicle_class}")
                    cv2.imwrite('data/outputs/fallback_crop.jpg', fallback_crop)
                    
                    plate_color, color_conf, _ = self.color_classifier.classify(fallback_crop)
                    print(f"Fallback color: {plate_color} | conf={color_conf:.2f}")
                    
                    enhanced_fallback    = self.enhancer.enhance(fallback_crop)
                    reg_number, ocr_conf = self.ocr_engine.extract(enhanced_fallback)
                    
                    if reg_number or (plate_color != PlateColor.UNKNOWN and color_conf > 0.20):
                        print(f"Fallback success: color={plate_color} reg={reg_number}")
                        confidence_result = self.confidence_engine.evaluate(
                            plate_color=plate_color,
                            color_conf=color_conf,
                            api_result=None,
                            cable_detected=False,
                            registration_number=reg_number
                        )
                        self.detection_logger.log(det, confidence_result)
                        frame = self.annotator.draw_result(frame, det, confidence_result)
                        results.append(confidence_result)
                        extracted = True
                        break
                if not extracted:
                    frame = self.annotator.draw_no_plate(frame, det)
                continue
            det.plate_bbox = plate_bbox
            det.plate_crop = plate_crop
            det.plate_conf = plate_conf

            # Stage 3: color classification
            plate_color, color_conf, color_scores = self.color_classifier.classify(plate_crop)

            # Stage 4: enhance plate crop + OCR
            reg_number = ""
            ocr_conf   = 0.0
            if plate_color != PlateColor.UNKNOWN:
                enhanced_crop = self.enhancer.enhance(plate_crop)
                cv2.imwrite('data/outputs/pipeline_plate_crop.jpg', enhanced_crop)
                print(f"Enhanced crop shape: {enhanced_crop.shape}")
                reg_number, ocr_conf = self.ocr_engine.extract(enhanced_crop)

            # Stage 5: API verification — disabled for POC phase 1
            api_result = None

            # Stage 6: confidence scoring
            confidence_result = self.confidence_engine.evaluate(
                plate_color=plate_color,
                color_conf=color_conf,
                api_result=api_result,
                cable_detected=False,
                registration_number=reg_number
            )

            if det.track_id:
                self._processed_tracks[det.track_id] = confidence_result

            self.detection_logger.log(det, confidence_result)
            frame = self.annotator.draw_result(frame, det, confidence_result)
            results.append(confidence_result)

        return frame, results

    def process_image(self, image_path: str):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.process_frame(frame)

    def process_video(self, video_path: str, output_path: Optional[str] = None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % settings.FRAME_SKIP != 0:
                if writer:
                    writer.write(frame)
                continue

            annotated, results = self.process_frame(frame)
            if writer:
                writer.write(annotated)
            yield annotated, results

        cap.release()
        if writer:
            writer.release()

    def reset_tracks(self):
        self._processed_tracks.clear()
        self._frame_count = 0