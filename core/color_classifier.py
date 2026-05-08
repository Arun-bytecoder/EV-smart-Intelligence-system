import cv2
import numpy as np
from config.settings import settings
from enum import Enum


class PlateColor(str, Enum):
    GREEN   = "green"
    WHITE   = "white"
    YELLOW  = "yellow"
    RED     = "red"
    UNKNOWN = "unknown"


class ColorClassifier:

    def classify(self, plate_crop: np.ndarray):
        if plate_crop is None or plate_crop.size == 0:
            return PlateColor.UNKNOWN, 0.0, {}

        hsv = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2HSV)
        total = plate_crop.shape[0] * plate_crop.shape[1]

        scores = {
            PlateColor.GREEN:  self._green_score(hsv, total),
            PlateColor.WHITE:  self._white_score(hsv, total),
            PlateColor.YELLOW: self._yellow_score(hsv, total),
            PlateColor.RED:    self._red_score(hsv, total),
        }

        best_color = max(scores, key=scores.get)
        best_score = scores[best_color]

        if best_score < 0.20:
            return PlateColor.UNKNOWN, best_score, scores

        return best_color, best_score, scores

    def _green_score(self, hsv, total):
        s = settings
        mask = cv2.inRange(
            hsv,
            (s.GREEN_H_MIN, s.GREEN_S_MIN, s.GREEN_V_MIN),
            (s.GREEN_H_MAX, s.GREEN_S_MAX, s.GREEN_V_MAX)
        )
        return cv2.countNonZero(mask) / total

    def _white_score(self, hsv, total):
        s = settings
        mask = cv2.inRange(
            hsv,
            (0,   0,          s.WHITE_V_MIN),
            (179, s.WHITE_S_MAX, 255)
        )
        return cv2.countNonZero(mask) / total

    def _yellow_score(self, hsv, total):
        s = settings
        mask = cv2.inRange(
            hsv,
            (s.YELLOW_H_MIN, 80, 100),
            (s.YELLOW_H_MAX, 255, 255)
        )
        return cv2.countNonZero(mask) / total

    def _red_score(self, hsv, total):
        s = settings
        mask1 = cv2.inRange(hsv, (s.RED_H_MIN1, 70, 50), (s.RED_H_MAX1, 255, 255))
        mask2 = cv2.inRange(hsv, (s.RED_H_MIN2, 70, 50), (s.RED_H_MAX2, 255, 255))
        mask  = cv2.bitwise_or(mask1, mask2)
        return cv2.countNonZero(mask) / total