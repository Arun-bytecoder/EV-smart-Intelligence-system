import requests
from dataclasses import dataclass
from typing import Optional
from config.settings import settings


@dataclass
class VehicleAPIResult:
    registration_number: str
    is_ev: Optional[bool]
    owner_name: Optional[str] = None
    vehicle_model: Optional[str] = None
    fuel_type: Optional[str] = None
    vehicle_class: Optional[str] = None
    registration_date: Optional[str] = None
    registration_state: Optional[str] = None
    raw_response: Optional[dict] = None
    error: Optional[str] = None


class VahanClient:

    def __init__(self):
        self.api_url = settings.VAHAN_API_URL
        self.api_key = settings.VAHAN_API_KEY
        self.timeout = 5

    def verify(self, registration_number: str) -> VehicleAPIResult:
        if not registration_number:
            return VehicleAPIResult(
                registration_number=registration_number,
                is_ev=None,
                error="Empty registration number"
            )

        if not self.api_key:
            print("No Vahan API key configured — using mock response.")
            return self._mock_response(registration_number)

        try:
            response = requests.get(
                self.api_url,
                params={"reg_no": registration_number},
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(registration_number, data)

        except requests.exceptions.Timeout:
            return VehicleAPIResult(
                registration_number=registration_number,
                is_ev=None,
                error="API timeout"
            )
        except requests.exceptions.RequestException as e:
            return VehicleAPIResult(
                registration_number=registration_number,
                is_ev=None,
                error=str(e)
            )

    def _parse_response(self, reg_no: str, data: dict) -> VehicleAPIResult:
        fuel_type = data.get("fuel_type", "").upper()
        is_ev     = fuel_type in ("ELECTRIC", "EV", "BATTERY")

        return VehicleAPIResult(
            registration_number=reg_no,
            is_ev=is_ev,
            owner_name=data.get("owner_name"),
            vehicle_model=data.get("vehicle_model") or data.get("maker_model"),
            fuel_type=fuel_type,
            vehicle_class=data.get("vehicle_class"),
            registration_date=data.get("registration_date"),
            registration_state=data.get("reg_authority"),
            raw_response=data
        )

    def _mock_response(self, reg_no: str) -> VehicleAPIResult:
        ev_states = ("TN", "KA", "MH", "DL", "GJ")
        is_ev     = reg_no[:2] in ev_states

        return VehicleAPIResult(
            registration_number=reg_no,
            is_ev=is_ev,
            owner_name="Mock Owner (API not configured)",
            vehicle_model="Tata Nexon EV" if is_ev else "Maruti Swift",
            fuel_type="ELECTRIC" if is_ev else "PETROL",
            vehicle_class="LMV",
            registration_date="2023-01-01",
            registration_state=reg_no[:2],
            error="Mock data — configure VAHAN_API_KEY in .env"
        )