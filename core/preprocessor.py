import cv2
import numpy as np
from config.settings import settings


class Preprocessor:

    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        gamma = Preprocessor._auto_gamma(frame)
        frame = Preprocessor._apply_gamma(frame, gamma)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        return frame

    @staticmethod
    def _auto_gamma(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 60:
            return 1.8
        elif mean_brightness < 110:
            return 1.3
        else:
            return 1.0

    @staticmethod
    def _apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(frame, table)

    @staticmethod
    def prepare_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        if h < settings.MIN_PLATE_HEIGHT:
            scale = settings.MIN_PLATE_HEIGHT / h
            plate_crop = cv2.resize(
                plate_crop,
                (int(w * scale), settings.MIN_PLATE_HEIGHT),
                interpolation=cv2.INTER_CUBIC
            )

        plate_crop = Preprocessor._deskew(plate_crop)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(binary, -1, kernel)

        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if abs(angle) < 15:
                    angles.append(angle)

            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    h, w = image.shape[:2]
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
                    image = cv2.warpAffine(image, M, (w, h),
                                           flags=cv2.INTER_CUBIC,
                                           borderMode=cv2.BORDER_REPLICATE)
        return image