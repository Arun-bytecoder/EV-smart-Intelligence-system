import json
from datetime import datetime
from pathlib import Path
from config.settings import settings


class DetectionLogger:

    def __init__(self):
        self.log_dir  = Path(settings.LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"detections_{datetime.now().strftime('%Y%m%d')}.jsonl"
        print(f"Detection log: {self.log_file}")

    def log(self, det, result):
        event = {
            "timestamp":           datetime.now().isoformat(),
            "track_id":            det.track_id,
            "vehicle_class":       det.vehicle_class,
            "vehicle_conf":        round(det.vehicle_conf, 3),
            "plate_color":         result.plate_color.value if result.plate_color else None,
            "color_confidence":    round(result.color_confidence, 3),
            "registration_number": result.registration_number,
            "decision":            result.decision.value,
            "score":               result.score,
            "flags":               result.flags,
            "alert_level":         result.alert_level,
            "owner_name":          result.owner_name,
            "vehicle_model":       result.vehicle_model,
            "fuel_type":           result.fuel_type,
            "api_is_ev":           result.api_result.is_ev if result.api_result else None,
            "api_error":           result.api_result.error if result.api_result else None,
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

        print(f"Event logged | decision={event['decision']} | reg={event['registration_number']}")