from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "alarm_events.csv"
MODEL_PATH = ROOT / "models" / "false_alarm_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
CONFUSION_PATH = ROOT / "reports" / "confusion_matrix.png"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.png"

TARGET = "is_real_alarm"  # 1 = real alarm, 0 = false alarm
RANDOM_STATE = 42

# Readings at the moment the alarm fired.
SENSOR_FEATURES = [
    "co2_ppm",
    "co_ppm",
    "humidity_pct",
    "temperature_c",
    "smoke_density",
    "pressure_hpa",
    "vibration_mm_s",
    "noise_db",
]
# How the readings moved in the 10 minutes before the alarm.
TREND_FEATURES = [
    "co2_change_10min",
    "temp_change_10min",
    "humidity_change_10min",
    "smoke_change_10min",
]
CONTEXT_FEATURES = [
    "alarm_duration_s",       # how long the sensor stayed above threshold
    "sensors_in_alarm",       # how many sensors exceeded their threshold at once
    "hour_of_day",
    "machine_running",        # 1 if nearby equipment was running
    "maintenance_active",     # 1 if maintenance/welding/cleaning was under way
]
NUMERIC_FEATURES = SENSOR_FEATURES + TREND_FEATURES + [
    "alarm_duration_s", "sensors_in_alarm", "hour_of_day",
]
BINARY_FEATURES = ["machine_running", "maintenance_active"]
ALARM_TYPES = ["smoke", "gas", "temperature", "humidity"]
CATEGORICAL_FEATURES = ["alarm_type"]  # one of ALARM_TYPES
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
