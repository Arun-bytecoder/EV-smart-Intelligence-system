from dataclasses import dataclass, field
from typing import Optional
from config.settings import settings
from core.color_classifier import PlateColor
from enum import Enum


class EVDecision(str, Enum):
    CONFIRMED_EV   = "confirmed_ev"
    HIGH_CONF_EV   = "high_conf_ev"
    UNCERTAIN      = "uncertain"
    NON_EV         = "non_ev"
    FRAUD_ALERT    = "fraud_alert"
    UNVERIFIED_EV  = "unverified_ev"


@dataclass
class ConfidenceResult:
    decision: EVDecision
    score: int
    flags: list = field(default_factory=list)

    plate_color: Optional[PlateColor] = None
    color_confidence: float = 0.0
    api_result: Optional[object] = None
    cable_detected: bool = False

    registration_number: str = ""
    owner_name: Optional[str] = None
    vehicle_model: Optional[str] = None
    fuel_type: Optional[str] = None

    summary: str = ""
    alert_level: str = "none"


class ConfidenceEngine:

    def evaluate(
        self,
        plate_color: PlateColor,
        color_conf: float,
        api_result=None,
        cable_detected: bool = False,
        registration_number: str = ""
        ) -> ConfidenceResult:
        
        score = 0
        flags = []

        # Signal 1: plate color
        if plate_color == PlateColor.GREEN:
            score = 2
            flags.append("green_plate_detected")
            decision = EVDecision.HIGH_CONF_EV
            alert_level = "none"
        elif plate_color in (PlateColor.WHITE, PlateColor.YELLOW, PlateColor.RED):
            score = 0
            flags.append(f"{plate_color.value}_plate_detected")
            decision = EVDecision.NON_EV
            alert_level = "critical"
        else:
            score = 0
            flags.append("plate_color_unknown")
            decision = EVDecision.UNCERTAIN
            alert_level = "warning"
            
        summary = self._build_summary(decision, registration_number, plate_color)
        
        return ConfidenceResult(
            decision=decision,
            score=score,
            flags=flags,
            plate_color=plate_color,
            color_confidence=color_conf,
            api_result=None,
            cable_detected=cable_detected,
            registration_number=registration_number,
            summary=summary,
            alert_level=alert_level
        )

    def _decide(self, score, plate_color, api_result, is_fraud) -> EVDecision:
        if is_fraud:
            return EVDecision.FRAUD_ALERT
        if score >= settings.SCORE_CONFIRMED_EV:
            return EVDecision.CONFIRMED_EV
        if score >= settings.SCORE_HIGH_CONF_EV:
            if plate_color == PlateColor.GREEN and (api_result is None or api_result.is_ev is None):
                return EVDecision.UNVERIFIED_EV
            return EVDecision.HIGH_CONF_EV
        if score >= settings.SCORE_UNCERTAIN:
            return EVDecision.UNCERTAIN
        return EVDecision.NON_EV

    def _alert_level(self, decision: EVDecision) -> str:
        return {
            EVDecision.CONFIRMED_EV:  "none",
            EVDecision.HIGH_CONF_EV:  "none",
            EVDecision.UNVERIFIED_EV: "warning",
            EVDecision.UNCERTAIN:     "warning",
            EVDecision.NON_EV:        "critical",
            EVDecision.FRAUD_ALERT:   "critical",
        }.get(decision, "warning")

    def _build_summary(self, decision, reg_no, plate_color) -> str:
        return {
            EVDecision.HIGH_CONF_EV:
                f"EV Detected — Green plate | Reg: {reg_no}",
            EVDecision.NON_EV:
                f"Non-EV Detected — {plate_color.value.title()} plate | Bay Alert!",
            EVDecision.UNCERTAIN:
                f"Plate color unclear — Manual check required",
        }.get(decision, "Unknown")