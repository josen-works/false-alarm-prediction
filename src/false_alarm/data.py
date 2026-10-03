"""Synthetic factory alarm data.

Real plant data is rarely shareable, so this module simulates it. Normal readings
drift around a baseline. A *real* alarm shows several correlated, sustained
anomalies (fire: temperature + smoke + CO2 + CO rise together; gas leak: CO2/CO
with a pressure drop). A *false* alarm is usually a single-sensor excursion that
is short, or explained by context (steam raising humidity, welding or cleaning
producing smoke, a door opening spiking CO2).

Swap `generate_dataset` for your own loader (same columns as config.ALL_FEATURES
plus config.TARGET) to train on real data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ALARM_TYPES = config.ALARM_TYPES


def _baseline(rng: np.random.Generator, n: int) -> pd.DataFrame:
    hour = rng.integers(0, 24, n)
    running = (rng.random(n) < np.where((hour >= 6) & (hour < 22), 0.85, 0.2)).astype(int)
    return pd.DataFrame({
        "co2_ppm": rng.normal(550 + 120 * running, 60, n),
        "co_ppm": np.abs(rng.normal(3 + 2 * running, 1.5, n)),
        "humidity_pct": np.clip(rng.normal(45, 8, n), 15, 95),
        "temperature_c": rng.normal(23 + 3 * running, 2, n),
        "smoke_density": np.abs(rng.normal(0.02, 0.01, n)),
        "pressure_hpa": rng.normal(1013, 3, n),
        "vibration_mm_s": np.abs(rng.normal(1 + 3 * running, 0.8, n)),
        "noise_db": rng.normal(55 + 20 * running, 4, n),
        "co2_change_10min": rng.normal(0, 20, n),
        "temp_change_10min": rng.normal(0, 0.4, n),
        "humidity_change_10min": rng.normal(0, 1.5, n),
        "smoke_change_10min": rng.normal(0, 0.005, n),
        "alarm_duration_s": rng.integers(2, 60, n).astype(float),
        "sensors_in_alarm": np.ones(n),
        "hour_of_day": hour,
        "machine_running": running,
        "maintenance_active": (rng.random(n) < 0.12).astype(int),
    })


def _make_real(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    kind = rng.choice(["fire", "gas_leak", "overheat"], n, p=[0.45, 0.3, 0.25])
    fire, gas, heat = kind == "fire", kind == "gas_leak", kind == "overheat"

    df.loc[fire, "temperature_c"] += rng.normal(25, 8, fire.sum())
    df.loc[fire, "smoke_density"] += rng.normal(0.5, 0.15, fire.sum())
    df.loc[fire, "co2_ppm"] += rng.normal(900, 250, fire.sum())
    df.loc[fire, "co_ppm"] += rng.normal(30, 15, fire.sum())
    df.loc[fire, "humidity_pct"] -= rng.normal(8, 4, fire.sum())

    df.loc[gas, "co2_ppm"] += rng.normal(1500, 400, gas.sum())
    df.loc[gas, "co_ppm"] += rng.normal(18, 10, gas.sum())
    df.loc[gas, "pressure_hpa"] -= rng.normal(6, 2, gas.sum())

    df.loc[heat, "temperature_c"] += rng.normal(30, 6, heat.sum())
    df.loc[heat, "vibration_mm_s"] += rng.normal(6, 2, heat.sum())
    df.loc[heat, "noise_db"] += rng.normal(12, 4, heat.sum())

    df["co2_change_10min"] += np.where(fire | gas, rng.normal(320, 160, n), 0)
    df["temp_change_10min"] += np.where(fire | heat, rng.normal(6, 3, n), 0)
    df["smoke_change_10min"] += np.where(fire, rng.normal(0.2, 0.1, n), 0)
    # Real events tend to persist, but a detector near the source can also trip briefly.
    df["alarm_duration_s"] = np.clip(rng.lognormal(np.log(120), 0.9, n), 5, 1800).round()
    df["sensors_in_alarm"] = np.where(fire, rng.integers(2, 5, n),
                             np.where(gas, rng.integers(2, 4, n), rng.integers(1, 3, n)))
    df["maintenance_active"] = (rng.random(n) < 0.05).astype(int)
    df["alarm_type"] = np.where(fire, "smoke", np.where(gas, "gas", "temperature"))
    return df


def _make_false(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    cause = rng.choice(["steam", "welding", "door_open", "sensor_glitch", "cleaning"],
                       n, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    steam, weld, door, glitch, clean = (cause == c for c in
                                        ["steam", "welding", "door_open", "sensor_glitch", "cleaning"])

    df.loc[steam, "humidity_pct"] = np.clip(rng.normal(88, 5, steam.sum()), 60, 100)
    df.loc[steam, "temperature_c"] += rng.normal(5, 2, steam.sum())
    df.loc[steam, "smoke_density"] += rng.normal(0.18, 0.06, steam.sum())

    df.loc[weld, "smoke_density"] += rng.normal(0.35, 0.1, weld.sum())
    df.loc[weld, "co_ppm"] += rng.normal(12, 6, weld.sum())
    df.loc[weld, "maintenance_active"] = 1

    df.loc[door, "co2_ppm"] += rng.normal(600, 200, door.sum())
    df.loc[door, "temp_change_10min"] += rng.normal(-1.5, 0.5, door.sum())
    df.loc[door, "co_ppm"] += rng.normal(10, 6, door.sum())  # forklift exhaust at loading doors

    df.loc[glitch, "co2_ppm"] += rng.choice([-1, 1], glitch.sum()) * rng.normal(1500, 500, glitch.sum())
    df.loc[glitch, "co2_ppm"] = df.loc[glitch, "co2_ppm"].clip(lower=0)

    df.loc[clean, "humidity_pct"] = np.clip(rng.normal(80, 6, clean.sum()), 50, 100)
    df.loc[clean, "co_ppm"] += rng.normal(5, 2, clean.sum())
    df.loc[clean, "maintenance_active"] = 1

    # Nuisance sources also move readings in the minutes before the trip.
    df.loc[steam, "humidity_change_10min"] += rng.normal(18, 6, steam.sum())
    df.loc[steam, "temp_change_10min"] += rng.normal(2.5, 1.5, steam.sum())
    df.loc[weld, "smoke_change_10min"] += rng.normal(0.12, 0.06, weld.sum())
    df.loc[weld, "temp_change_10min"] += rng.normal(2, 1.5, weld.sum())
    df.loc[door, "co2_change_10min"] += rng.normal(250, 120, door.sum())
    df.loc[glitch, "co2_change_10min"] += rng.choice([-1, 1], glitch.sum()) * rng.normal(600, 300, glitch.sum())
    df.loc[clean, "humidity_change_10min"] += rng.normal(12, 5, clean.sum())

    # Most nuisance trips are short, but steam and cleaning can keep a sensor tripped for minutes.
    long_cause = steam | clean
    df["alarm_duration_s"] = np.clip(
        rng.lognormal(np.log(np.where(long_cause, 90, 25)), 0.9, n), 1, 1800).round()
    df["sensors_in_alarm"] = np.where(rng.random(n) < 0.88, 1, 2)
    df["alarm_type"] = np.where(steam | clean, "humidity",
                       np.where(weld, "smoke", np.where(door | glitch, "gas", "temperature")))
    return df


def generate_dataset(n_samples: int = 5000, real_ratio: float = 0.2,
                     label_noise: float = 0.02, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Create a labelled alarm-event table (real alarms are the minority class)."""
    rng = np.random.default_rng(seed)
    n_real = int(n_samples * real_ratio)
    real = _make_real(_baseline(rng, n_real), rng)
    false = _make_false(_baseline(rng, n_samples - n_real), rng)
    real[config.TARGET] = 1
    false[config.TARGET] = 0
    df = pd.concat([real, false], ignore_index=True)

    flip = rng.random(len(df)) < label_noise  # mislabelled events, as in real logs
    df.loc[flip, config.TARGET] = 1 - df.loc[flip, config.TARGET]

    df["pressure_hpa"] = df["pressure_hpa"].round(1)
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["sensors_in_alarm"] = df["sensors_in_alarm"].astype(int)
    df = df.round(3).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[config.ALL_FEATURES + [config.TARGET]]


def load_dataset(path=config.DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. Run: python -m false_alarm generate")
    df = pd.read_csv(path)
    missing = [c for c in config.ALL_FEATURES + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return df
