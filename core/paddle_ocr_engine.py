import cv2
import numpy as np
import re
from paddleocr import PaddleOCR


OCR_CORRECTIONS = {
    "O": "0", "I": "1", "B": "8",
    "S": "5", "Z": "2", "G": "6",
}

PLATE_PATTERN = re.compile(
    r"([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{1,3})\s*([0-9]{4})"
)


class PaddleOCREngine:

    def __init__(self):
        print("Initializing PaddleOCR engine...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False,
            rec_algorithm='SVTR_LCNet',   # best accuracy model
        )
        print("PaddleOCR engine ready.")

    def extract(self, plate_crop: np.ndarray):
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # Upscale for better accuracy
        h, w = plate_crop.shape[:2]
        if h < 60:
            scale = 60 / h
            plate_crop = cv2.resize(
                plate_crop,
                (int(w * scale), 60),
                interpolation=cv2.INTER_LANCZOS4
            )

        try:
            result = self.ocr.ocr(plate_crop, cls=True)
            if not result or not result[0]:
                return "", 0.0

            texts = []
            confs = []
            for line in result[0]:
                text = line[1][0].upper()
                conf = line[1][1]
                texts.append(text)
                confs.append(conf)
                print(f"PaddleOCR: '{text}' | conf={conf:.2f}")

            combined = "".join(texts)
            avg_conf = sum(confs) / len(confs)

            cleaned    = re.sub(r"[^A-Z0-9]", "", combined)
            reg_number = self._match_and_correct(cleaned)

            if reg_number:
                print(f"Extracted: {reg_number} | conf={avg_conf:.2f}")
                return reg_number, avg_conf

            return cleaned, avg_conf * 0.5

        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return "", 0.0

    def _match_and_correct(self, text: str) -> str:
        # Direct match
        match = PLATE_PATTERN.search(text)
        if match:
            return "".join(match.groups())

        # Positional correction
        corrected = self._positional_correction(text)
        match = PLATE_PATTERN.search(corrected)
        if match:
            return "".join(match.groups())

        return ""

    def _positional_correction(self, text: str) -> str:
        if len(text) < 9:
            return text

        result = list(text)

        digit_to_letter = {"0":"O","1":"I","8":"B","5":"S","2":"Z","4":"A"}
        letter_to_digit = {"O":"0","I":"1","B":"8","S":"5","Z":"2","A":"4","G":"6","D":"0"}

        # State code (0-1): letters
        for i in range(min(2, len(result))):
            if result[i].isdigit():
                result[i] = digit_to_letter.get(result[i], result[i])

        # District (2-3): digits
        for i in range(2, min(4, len(result))):
            if result[i].isalpha():
                result[i] = letter_to_digit.get(result[i], result[i])

        # Series (4 to -4): letters
        for i in range(4, max(4, len(result)-4)):
            if result[i].isdigit():
                result[i] = digit_to_letter.get(result[i], result[i])

        # Number (-4 to end): digits
        for i in range(max(0, len(result)-4), len(result)):
            if result[i].isalpha():
                result[i] = letter_to_digit.get(result[i], result[i])

        return "".join(result)