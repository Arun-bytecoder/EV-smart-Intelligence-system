from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:

    # --- Model paths ---
    VEHICLE_MODEL_PATH = str(BASE_DIR / "models" / "yolov8n.pt")
    PLATE_MODEL_PATH   = str(BASE_DIR / "models" / "plate_detector.pt")

    # --- Detection thresholds ---
    VEHICLE_CONF_THRESHOLD = 0.35
    PLATE_CONF_THRESHOLD   = 0.35
    OCR_MIN_CONFIDENCE     = 0.55

    # --- HSV color ranges ---
    GREEN_H_MIN = 25;  GREEN_H_MAX = 100
    GREEN_S_MIN = 25;  GREEN_S_MAX = 255
    GREEN_V_MIN = 15;  GREEN_V_MAX = 220


    WHITE_S_MAX  = 30;  WHITE_V_MIN = 220
    YELLOW_H_MIN = 20;  YELLOW_H_MAX = 35
    RED_H_MIN1   = 0;   RED_H_MAX1   = 10
    RED_H_MIN2   = 160; RED_H_MAX2   = 179

    # --- OCR ---
    OCR_ENGINE    = "easyocr"
    OCR_LANGUAGES = ["en"]

    # --- Indian plate regex ---
    PLATE_REGEX = r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}"

    # --- Frame processing ---
    FRAME_SKIP       = 3
    MIN_PLATE_HEIGHT = 200

    # --- Vahan API ---
    VAHAN_API_URL = os.getenv("VAHAN_API_URL", "https://apisetu.gov.in/vahan/v3/rc/findByRegNo")
    VAHAN_API_KEY = os.getenv("VAHAN_API_KEY", "")

    # --- Confidence scoring ---
    SCORE_GREEN_PLATE    =  1
    SCORE_API_EV         =  1
    SCORE_API_NON_EV     = -1
    SCORE_CABLE_DETECTED =  1

    SCORE_CONFIRMED_EV  = 3
    SCORE_HIGH_CONF_EV  = 2
    SCORE_UNCERTAIN     = 1

    # --- Logging ---
    LOG_DIR   = str(BASE_DIR / "data" / "logs")
    LOG_LEVEL = "INFO"

    # --- Output ---
    OUTPUT_DIR = str(BASE_DIR / "data" / "outputs")

settings = Settings()