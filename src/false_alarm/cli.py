import argparse
import json
from pathlib import Path

import pandas as pd

from . import config
from .data import generate_dataset, load_dataset
from .predict import predict_events
from .train import train


def main(argv=None):
    p = argparse.ArgumentParser(prog="false_alarm", description="Factory false-alarm classifier")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a synthetic alarm dataset")
    g.add_argument("--samples", type=int, default=5000)
    g.add_argument("--real-ratio", type=float, default=0.2)
    g.add_argument("--out", default=str(config.DATA_PATH))

    t = sub.add_parser("train", help="train, compare and save the best model")
    t.add_argument("--data", default=str(config.DATA_PATH))

    pr = sub.add_parser("predict", help="classify events from a CSV, or one event via --event JSON")
    pr.add_argument("--csv")
    pr.add_argument("--event", help='JSON, e.g. \'{"co2_ppm": 2400, ...}\' (all features required)')
    pr.add_argument("--out")

    a = p.parse_args(argv)
    try:
        _run(a, p)
    except (FileNotFoundError, ValueError) as e:  # JSONDecodeError is a ValueError
        p.exit(1, f"error: {e}\n")


def _run(a, p):
    if a.cmd == "generate":
        df = generate_dataset(a.samples, a.real_ratio)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"Wrote {len(df)} events ({df[config.TARGET].mean():.1%} real) to {a.out}")
    elif a.cmd == "train":
        train(load_dataset(a.data))
    else:
        if a.event:
            try:
                events = pd.DataFrame([json.loads(a.event)])
            except json.JSONDecodeError as e:
                raise ValueError(f"--event is not valid JSON ({e.msg}); use double quotes around keys") from None
        elif a.csv:
            if not Path(a.csv).exists():
                raise FileNotFoundError(f"no such file: {a.csv}")
            events = pd.read_csv(a.csv)
        else:
            p.error("give --csv or --event")
        result = predict_events(events)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(a.out, index=False)
            print(f"Wrote {len(result)} predictions to {a.out}")
        cols = ["real_alarm_probability", "verdict"]
        if len(result) <= 20:
            print(result[cols].to_string())
        else:
            counts = result["verdict"].value_counts()
            print(f"{len(result)} events: {counts.get('REAL', 0)} REAL, {counts.get('FALSE', 0)} FALSE")
            print(result[cols].head(10).to_string())
            if not a.out:
                print("... (use --out FILE to save every prediction)")


if __name__ == "__main__":
    main()
