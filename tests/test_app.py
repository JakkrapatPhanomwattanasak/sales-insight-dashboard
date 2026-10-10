from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

import data_app
from data_app import create_app
from data_app.pipeline import analyze


@pytest.fixture
def client():
    return create_app({"TESTING": True}).test_client()


def demo_bytes():
    return (Path(data_app.__file__).parent / "data/demo_sales.csv").read_bytes()


def upload(client, content):
    return client.post("/api/analyze", data={"file": (BytesIO(content), "sales.csv")})


def test_complete_demo_flow(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/report").status_code == 409
    report = client.post("/api/demo").json
    assert report["rows"] == 120
    assert report["metrics"]["mae"] < report["metrics"]["baseline_mae"]
    assert client.get("/api/report").json == report
    result = client.post(
        "/api/predict",
        json={"marketing_spend": 1000, "visits": 500, "region": "Central", "channel": "Online"},
    )
    assert result.status_code == 200 and result.json["predicted_revenue"] > 0


def test_deduplication_and_missing_feature_imputation(client):
    frame = pd.read_csv(BytesIO(demo_bytes()))
    frame.loc[0, "marketing_spend"] = float("nan")
    frame = pd.concat([frame, frame.iloc[[1]]])
    response = upload(client, frame.to_csv(index=False).encode())
    assert response.status_code == 200
    assert response.json["duplicates_removed"] == 1
    assert response.json["missing_numeric_values"] == 1


@pytest.mark.parametrize("content", [b"", b"a,b\n1,2", b"\xff\xfe", b"x" * 100])
def test_invalid_csv_does_not_replace_existing_analysis(client, content):
    expected = client.post("/api/demo").json
    assert upload(client, content).status_code == 400
    assert client.get("/api/report").json == expected


@pytest.mark.parametrize(
    "field,value",
    [("revenue", -1), ("visits", "invalid"), ("region", ""), ("revenue", float("inf"))],
)
def test_invalid_values(client, field, value):
    frame = pd.read_csv(BytesIO(demo_bytes()))
    frame[field] = frame[field].astype(object)
    frame.loc[0, field] = value
    assert upload(client, frame.to_csv(index=False).encode()).status_code == 400


def test_prediction_validation_and_limits(client):
    assert client.post("/api/predict", json={}).status_code == 409
    client.post("/api/demo")
    valid = {"marketing_spend": 1000, "visits": 500, "region": "North", "channel": "Retail"}
    for field, value in [("visits", True), ("region", "unknown"), ("visits", -1)]:
        assert client.post("/api/predict", json={**valid, field: value}).status_code == 400
    assert client.post("/api/predict", json={**valid, "extra": 1}).status_code == 400
    assert upload(client, b"x" * (5 * 1024 * 1024)).status_code == 413


def test_reproducible_holdout_metrics():
    assert analyze(demo_bytes()).report["metrics"] == analyze(demo_bytes()).report["metrics"]
