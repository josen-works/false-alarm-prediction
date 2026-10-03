from fastapi.testclient import TestClient

from false_alarm import api, config
from false_alarm.data import generate_dataset
from false_alarm.predict import predict_events
from false_alarm.train import train


def test_dataset_shape_and_balance():
    df = generate_dataset(1000, real_ratio=0.2, seed=1)
    assert set(config.ALL_FEATURES + [config.TARGET]) == set(df.columns)
    assert 0.15 < df[config.TARGET].mean() < 0.25
    assert not df.isna().any().any()


def test_train_and_predict(tmp_path, monkeypatch):
    path = tmp_path / "m.joblib"
    metrics = train(generate_dataset(1500, seed=2), model_path=path, report=False)
    assert metrics["test_roc_auc"] > 0.9
    df = generate_dataset(50, seed=3)
    out = predict_events(df[config.ALL_FEATURES], path=path)
    assert set(out["verdict"]) <= {"REAL", "FALSE"}
    assert out["real_alarm_probability"].between(0, 1).all()


def test_api(tmp_path, monkeypatch):
    path = tmp_path / "m.joblib"
    df = generate_dataset(1500, seed=4)
    train(df, model_path=path, report=False)
    monkeypatch.setattr(config, "MODEL_PATH", path)
    monkeypatch.setattr(api, "load_model", lambda: __import__("false_alarm.predict", fromlist=["x"]).load_model(path))
    client = TestClient(api.app)
    event = df[config.ALL_FEATURES].iloc[0].to_dict()
    event = {k: (v.item() if hasattr(v, "item") else v) for k, v in event.items()}
    r = client.post("/predict", json=event)
    assert r.status_code == 200 and r.json()["verdict"] in ("REAL", "FALSE")


def test_validation_rejects_bad_events():
    import pytest

    from false_alarm.predict import validate_events

    good = generate_dataset(5, seed=5)[config.ALL_FEATURES]
    validate_events(good)
    with pytest.raises(ValueError, match="missing fields"):
        validate_events(good.drop(columns=["co2_ppm"]))
    with pytest.raises(ValueError, match="unknown alarm_type"):
        validate_events(good.assign(alarm_type="fire"))
    with pytest.raises(ValueError, match="empty values"):
        validate_events(good.assign(co2_ppm=None))


def test_cli_reports_errors_without_traceback(capsys):
    import pytest

    from false_alarm.cli import main

    with pytest.raises(SystemExit) as e:
        main(["predict", "--event", "{bad json"])
    assert e.value.code == 1
    assert "not valid JSON" in capsys.readouterr().err
