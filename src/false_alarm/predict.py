"""Load the saved model and classify alarm events."""
import joblib
import pandas as pd

from . import config

_cache = {}


def load_model(path=config.MODEL_PATH) -> dict:
    if path not in _cache:
        if not path.exists():
            raise FileNotFoundError(f"No model at {path}. Run: python -m false_alarm train")
        _cache[path] = joblib.load(path)
    return _cache[path]


def validate_events(events: pd.DataFrame, features=config.ALL_FEATURES):
    """Raise ValueError with a readable message if the events can't be scored."""
    missing = [c for c in features if c not in events.columns]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    empty = [c for c in features if events[c].isna().any()]
    if empty:
        raise ValueError(f"empty values in: {', '.join(empty)}")
    bad = sorted(set(events["alarm_type"].astype(str)) - set(config.ALARM_TYPES))
    if bad:
        raise ValueError(f"unknown alarm_type {bad}; expected one of {config.ALARM_TYPES}")
    numeric = [c for c in features if c != "alarm_type"]
    non_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(events[c])]
    if non_numeric:
        raise ValueError(f"non-numeric values in: {', '.join(non_numeric)}")


def predict_events(events: pd.DataFrame, path=config.MODEL_PATH) -> pd.DataFrame:
    """Return the events with `real_alarm_probability` and a `verdict` column."""
    bundle = load_model(path)
    validate_events(events, bundle["features"])
    proba = bundle["pipeline"].predict_proba(events[bundle["features"]])[:, 1]
    out = events.copy()
    out["real_alarm_probability"] = proba.round(4)
    out["verdict"] = ["REAL" if p >= bundle["threshold"] else "FALSE" for p in proba]
    return out
