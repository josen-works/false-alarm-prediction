"""REST API: `uvicorn false_alarm.api:app --reload` (docs at /docs)."""
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import config
from .predict import load_model

app = FastAPI(title="Factory False-Alarm Classifier")


EXAMPLE_EVENT = {  # a fire-like event; pre-filled in the /docs "Try it out" form
    "co2_ppm": 2400, "co_ppm": 45, "humidity_pct": 38, "temperature_c": 58,
    "smoke_density": 0.6, "pressure_hpa": 1012, "vibration_mm_s": 2, "noise_db": 60,
    "co2_change_10min": 500, "temp_change_10min": 9, "humidity_change_10min": -5,
    "smoke_change_10min": 0.3, "alarm_duration_s": 240, "sensors_in_alarm": 4,
    "hour_of_day": 14, "machine_running": 1, "maintenance_active": 0, "alarm_type": "smoke",
}


class AlarmEvent(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_EVENT]})

    co2_ppm: float
    co_ppm: float
    humidity_pct: float = Field(ge=0, le=100)
    temperature_c: float
    smoke_density: float
    pressure_hpa: float
    vibration_mm_s: float
    noise_db: float
    co2_change_10min: float
    temp_change_10min: float
    humidity_change_10min: float
    smoke_change_10min: float
    alarm_duration_s: float = Field(ge=0)
    sensors_in_alarm: int = Field(ge=1)
    hour_of_day: int = Field(ge=0, le=23)
    machine_running: int = Field(ge=0, le=1)
    maintenance_active: int = Field(ge=0, le=1)
    alarm_type: Literal["smoke", "gas", "temperature", "humidity"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict")
def predict(event: AlarmEvent):
    import pandas as pd
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    row = pd.DataFrame([event.model_dump()])[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {"real_alarm_probability": round(p, 4),
            "verdict": "REAL" if p >= bundle["threshold"] else "FALSE",
            "threshold": round(bundle["threshold"], 4)}
