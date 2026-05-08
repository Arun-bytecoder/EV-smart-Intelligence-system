import re
import numpy as np
import cv2
import easyocr
from config.settings import settings

# Character correction maps — applied based on position context
DIGIT_CORRECTIONS = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6", "D": "0"}
LETTER_CORRECTIONS = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}

# Indian plate: 2 letters + 2 digits + 1-3 letters + 4 digits = 10 chars
# Examples: TN11AW4253, MH04JM8765, KL08CB3215, DL3CAF1234
PLATE_PATTERN = re.compile(
    r"([A-Z]{2})\s*([0-9]{1,2})\s*([A-Z]{1,3})\s*([0-9]{4})"
)


class OCREngine:

    def __init__(self):
        print("Initializing OCR engine...")
        self.reader = easyocr.Reader(
            settings.OCR_LANGUAGES,
            gpu=False
        )
        print("OCR engine ready.")

    def extract(self, plate_crop: np.ndarray):
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # Try multiple preprocessing versions and pick best result
        candidates = self._get_candidates(plate_crop)

        best_reg    = ""
        best_conf   = 0.0

        for name, processed in candidates:
            raw_results = self._run_ocr(processed)
            if not raw_results:
                continue

            raw_text = " ".join([r[1] for r in raw_results])
            avg_conf = sum([r[2] for r in raw_results]) / len(raw_results)
            print(f"OCR [{name}]: '{raw_text}' | conf={avg_conf:.2f}")

            cleaned    = self._clean_text(raw_text)
            reg_number = self._match_plate_format(cleaned)

            if reg_number and avg_conf > best_conf:
                best_reg  = reg_number
                best_conf = avg_conf

        if best_reg:
            print(f"Registration extracted: {best_reg} | conf={best_conf:.2f}")
        else:
            print("Could not extract valid registration number.")

        return best_reg, best_conf

    def _get_candidates(self, crop: np.ndarray) -> list:
        candidates = []

        # Version 1: Raw crop upscaled
        h, w = crop.shape[:2]
        scale = max(1, 200 // h)
        upscaled = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        candidates.append(("upscaled", upscaled))

        # Version 2: Grayscale + CLAHE contrast boost
        gray  = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        candidates.append(("clahe", gray))

        # Version 3: Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("binary", binary))

        # Version 4: Inverted binary (white text on dark bg)
        inverted = cv2.bitwise_not(binary)
        candidates.append(("inverted", inverted))

        return candidates

    def _run_ocr(self, crop: np.ndarray):
        try:
            return self.reader.readtext(
                crop,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1,
                paragraph=False
            )
        except Exception as e:
            print(f"OCR error: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        text = text.upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def _match_plate_format(self, text: str) -> str:
        # Direct match
        match = PLATE_PATTERN.search(text)
        if match:
            result = "".join(match.groups())
            if len(result) >= 9:
                return result

        # Apply position-aware corrections
        corrected = self._positional_correction(text)
        match = PLATE_PATTERN.search(corrected)
        if match:
            result = "".join(match.groups())
            if len(result) >= 9:
                return result

        return ""

    def _positional_correction(self, text: str) -> str:
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(text) < 9:
            return text
        
        result = list(text)
        total  = len(result)
        
        # Map: digit that looks like a letter
        digit_to_letter = {
            "0": "O", "1": "I", "8": "B",
            "5": "S", "2": "Z", "4": "A"
        }
        
        # Map: letter that looks like a digit
        letter_to_digit = {
            "O": "0", "I": "1", "B": "8",
            "S": "5", "Z": "2", "A": "4",
            "G": "6", "D": "0"
        }

        # Position 0-1: STATE CODE — must be letters
        for i in range(min(2, total)):
            if result[i].isdigit():
                result[i] = digit_to_letter.get(result[i], result[i])

        # Position 2-3: DISTRICT CODE — must be digits
        for i in range(2, min(4, total)):
            if result[i].isalpha():
                result[i] = letter_to_digit.get(result[i], result[i])

        # Position 4 to total-4: SERIES — must be letters
        series_end = total - 4
        
        for i in range(4, max(4, series_end)):
            if result[i].isdigit():
                result[i] = digit_to_letter.get(result[i], result[i])

        # Last 4 positions: UNIQUE NUMBER — must be digits
        for i in range(max(0, total - 4), total):
            if result[i].isalpha():
                result[i] = letter_to_digit.get(result[i], result[i])
                
        corrected = "".join(result)
        print(f"Positional correction: {text} → {corrected}")
        return corrected