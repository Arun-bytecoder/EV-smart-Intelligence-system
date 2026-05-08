import cv2
import numpy as np
from typing import Optional

from core.preprocessor import Preprocessor
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
                # Fallback: try OCR directly on bottom 30% of vehicle crop
                x1, y1, x2, y2 = det.vehicle_bbox
                vh = y2 - y1
                # Crop bottom 35% of vehicle — where plate always is
                fallback_crop = enhanced[
                    y1 + int(vh * 0.65): y2,
                    x1: x2
                ]
                if fallback_crop.size > 0:
                    print("Plate detector failed — trying fallback bottom crop")
                    import cv2
                    cv2.imwrite('data/outputs/fallback_crop.jpg', fallback_crop)
                    # Try color classification on fallback
                    plate_color, color_conf, _ = self.color_classifier.classify(fallback_crop)
                    reg_number, ocr_conf = self.ocr_engine.extract(fallback_crop)
                    
                    if reg_number:
                        print(f"Fallback OCR extracted: {reg_number}")
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
                    else:
                        frame = self.annotator.draw_no_plate(frame, det)
                else:
                    frame = self.annotator.draw_no_plate(frame, det)
                continue

            det.plate_bbox = plate_bbox
            det.plate_crop = plate_crop
            det.plate_conf = plate_conf

            # Stage 2b: prepare plate crop
            prepared_crop = self.preprocessor.prepare_plate_crop(plate_crop)

            # Stage 3: color classification
            plate_color, color_conf, color_scores = self.color_classifier.classify(plate_crop)


            # Stage 4: OCR — run on RAW plate crop (best accuracy)
            # Stage 4: OCR — enhance crop before reading
            reg_number = ""
            ocr_conf   = 0.0
            if plate_color != PlateColor.UNKNOWN:
                import cv2
                # Aggressively upscale plate crop before OCR
                h, w = plate_crop.shape[:2]
                # Target minimum 60px height for OCR
                if h < 60:
                    scale = 60 / h
                    enhanced_crop = cv2.resize(
                        plate_crop,
                        (int(w * scale), 60),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                else:
                    enhanced_crop = plate_crop.copy()
                # Save for debug
                cv2.imwrite('data/outputs/pipeline_plate_crop.jpg', enhanced_crop)
                print(f"Pipeline plate crop shape: {enhanced_crop.shape}")
                reg_number, ocr_conf = self.ocr_engine.extract(enhanced_crop)
            
            # Stage 5: API verification
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