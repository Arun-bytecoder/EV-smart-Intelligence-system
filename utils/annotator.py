import cv2
import numpy as np
from core.confidence_engine import ConfidenceResult, EVDecision
from core.detector import DetectionResult

COLORS = {
    EVDecision.CONFIRMED_EV:  (0, 200, 0),
    EVDecision.HIGH_CONF_EV:  (0, 180, 0),
    EVDecision.UNVERIFIED_EV: (0, 200, 200),
    EVDecision.UNCERTAIN:     (0, 165, 255),
    EVDecision.NON_EV:        (0, 0, 220),
    EVDecision.FRAUD_ALERT:   (0, 0, 255),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:

    def draw_result(self, frame: np.ndarray, det, result: ConfidenceResult) -> np.ndarray:
        color = COLORS.get(result.decision, (128, 128, 128))
        x1, y1, x2, y2 = det.vehicle_bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_lines = [
            f"{result.decision.value.upper().replace('_', ' ')}",
            f"Plate: {result.plate_color.value if result.plate_color else 'N/A'}",
            f"Reg: {result.registration_number or 'Not extracted'}",
        ]
        if result.vehicle_model:
            label_lines.append(f"Model: {result.vehicle_model}")

        self._draw_label_box(frame, label_lines, (x1, y1), color)

        if det.plate_bbox:
            px1, py1, px2, py2 = det.plate_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 1)

        if result.alert_level == "critical":
            self._draw_alert_banner(frame, result.summary)

        return frame

    def draw_no_plate(self, frame: np.ndarray, det) -> np.ndarray:
        x1, y1, x2, y2 = det.vehicle_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(frame, "No plate detected", (x1, y1 - 8),
                    FONT, 0.5, (128, 128, 128), 1)
        return frame

    def draw_no_detection(self, frame: np.ndarray) -> np.ndarray:
        cv2.putText(frame, "No vehicle detected", (20, 40),
                    FONT, 0.7, (200, 200, 200), 2)
        return frame

    def _draw_label_box(self, frame, lines, origin, color):
        x, y    = origin
        line_h  = 20
        padding = 6
        max_w   = max(cv2.getTextSize(l, FONT, 0.45, 1)[0][0] for l in lines)
        box_h   = line_h * len(lines) + padding * 2
        y_start = max(y - box_h, 0)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y_start), (x + max_w + padding * 2, y), color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        for i, line in enumerate(lines):
            ty = y_start + padding + (i + 1) * line_h - 4
            cv2.putText(frame, line, (x + padding, ty), FONT, 0.45, (255, 255, 255), 1)

    def _draw_alert_banner(self, frame, message):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, message[:80], (10, h - 18),
                    FONT, 0.6, (255, 255, 255), 2)