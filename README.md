# ⚡ EV Smart Intelligence System — POC

Production-grade Proof of Concept for detecting EV vs Non-EV vehicles
at charging bays using multi-signal verification.

---

## Problem Solved

Non-EV vehicles park at EV charging bays (ICEing), blocking genuine EV users
who then waste battery range searching for alternatives. This system detects
and alerts in real-time — with fraud detection when a fake green plate is used.

---

## Detection Pipeline

```
Input (image / video / RTSP)
    ↓
[Stage 0] Frame preprocessing — CLAHE + gamma correction (day & night)
    ↓
[Stage 1] Vehicle detection — YOLOv8 + ByteTrack
    ↓
[Stage 2] License plate detection — YOLOv8 plate model
    ↓
[Stage 3] Plate color classification — HSV masking
           Green → possible EV | White/Yellow/Red → Non-EV
    ↓
[Stage 4] OCR — EasyOCR + regex post-processing
           Extracts Indian plate format: XX 00 XX 0000
    ↓
[Stage 5] Vahan API verification — cross-checks fuel type
    ↓
[Stage 6] Confidence scoring — 3-signal decision engine
           Score 3 = Confirmed EV | Score 0 = Non-EV | Fraud alert
    ↓
Dashboard — annotated frame + alert + vehicle details
```

---

## Multi-Signal Decision Matrix

| Plate Color | API Result  | Final Decision         |
|-------------|-------------|------------------------|
| Green       | EV          | ✅ Confirmed EV         |
| Green       | Non-EV      | 🚨 FRAUD — fake plate   |
| Green       | Unreachable | ⚠️ Unverified EV        |
| White/Yellow| Non-EV      | 🚨 Non-EV — Bay alert   |
| White/Yellow| EV          | ⚠️ Plate issue — review |
| Unknown     | —           | ⚠️ Operator check       |

---

## Project Structure

```
ev_smart_poc/
├── config/
│   └── settings.py          # All thresholds, paths, HSV ranges
├── core/
│   ├── preprocessor.py      # CLAHE, gamma, deskew
│   ├── detector.py          # YOLOv8 vehicle + plate detection
│   ├── color_classifier.py  # HSV plate color classification
│   ├── ocr_engine.py        # EasyOCR + regex post-processing
│   ├── confidence_engine.py # Multi-signal scoring + fraud detection
│   └── pipeline.py          # Main orchestrator
├── api/
│   └── vahan_client.py      # Vahan/Parivahan API integration
├── utils/
│   ├── annotator.py         # Frame annotation + alert overlay
│   └── logger.py            # Structured JSON event logging
├── dashboard/
│   └── app.py               # Streamlit POC dashboard
├── data/
│   ├── logs/                # Detection event logs (JSONL)
│   └── outputs/             # Annotated output frames/videos
├── models/                  # YOLOv8 model weights (download separately)
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
# Vehicle detection (YOLOv8n — pretrained COCO)
# Auto-downloads on first run via Ultralytics

# Plate detection model — use one of:
# Option A: https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
# Option B: Train custom on Indian plates dataset

# Place models in: models/
# Update paths in config/settings.py if needed

# 4. Configure API key
cp .env.example .env
# Edit .env and add your VAHAN_API_KEY
# Without key, system runs with mock API responses

# 5. Run dashboard
streamlit run dashboard/app.py
```

---

## Tech Stack

| Component       | Library              |
|----------------|----------------------|
| Detection       | YOLOv8 (Ultralytics) |
| Tracking        | ByteTrack (built-in) |
| Color classify  | OpenCV HSV masking   |
| OCR             | EasyOCR              |
| Preprocessing   | OpenCV + scikit-image|
| API             | Requests + Vahan API |
| Dashboard       | Streamlit            |
| Inference opt.  | ONNX Runtime         |
| Logging         | Loguru + JSONL       |

---

## Future Roadmap (Post-POC)

- [ ] Real-time RTSP stream support
- [ ] Cable / charging port detection model
- [ ] Mobile alert push notification
- [ ] Analytics dashboard (bay utilization, ICEing frequency)
- [ ] Hardware barrier integration (deny charger on Non-EV)
- [ ] Edge deployment on Jetson Nano / Raspberry Pi