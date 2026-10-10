"""CSV ingestion, validated preprocessing, regression, and report generation."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["marketing_spend", "visits"]
CATEGORICAL = ["region", "channel"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "revenue"


@dataclass
class Analysis:
    model: Pipeline
    report: dict


def read_sales(content):
    try:
        frame = pd.read_csv(BytesIO(content), encoding="utf-8")
    except (ValueError, UnicodeError) as error:
        raise ValueError("Upload a valid UTF-8 CSV with a header") from error
    if not set(FEATURES + [TARGET]).issubset(frame.columns):
        raise ValueError("CSV requires marketing_spend, visits, region, channel, revenue")
    if not 30 <= len(frame) <= 5000:
        raise ValueError("CSV must contain 30 to 5000 rows")
    frame = frame[FEATURES + [TARGET]].copy()
    original_rows = len(frame)
    for field in NUMERIC + [TARGET]:
        try:
            frame[field] = pd.to_numeric(frame[field], errors="raise")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{field} must be numeric") from error
        if np.isinf(frame[field]).any() or (frame[field].dropna() < 0).any():
            raise ValueError(f"{field} must contain finite, non-negative values")
    if frame[TARGET].isna().any() or frame[NUMERIC].isna().all().any():
        raise ValueError("Revenue is required and numeric features cannot be entirely missing")
    for field in CATEGORICAL:
        if frame[field].isna().any():
            raise ValueError(f"{field} is required")
        frame[field] = frame[field].astype(str).str.strip()
        if frame[field].eq("").any() or frame[field].str.len().gt(80).any():
            raise ValueError(f"{field} must contain 1 to 80 characters")
        if frame[field].nunique() > 50:
            raise ValueError(f"{field} supports at most 50 categories")
    frame = frame.drop_duplicates().reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError("Need at least 30 distinct records")
    if frame[TARGET].nunique() < 2:
        raise ValueError("Revenue must vary to evaluate a regression model")
    return frame, original_rows - len(frame)


def analyze(content):
    frame, duplicates = read_sales(content)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.2, random_state=42
    )
    if x_train[NUMERIC].isna().all().any():
        raise ValueError("Training split must contain observed values for each numeric feature")
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    transform = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    model = Pipeline([("features", transform), ("regressor", Ridge(alpha=1.0))])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train).predict(x_test)
    summary = frame.groupby("region")[TARGET].agg(["sum", "mean", "count"]).reset_index()
    report = {
        "rows": len(frame),
        "duplicates_removed": duplicates,
        "missing_numeric_values": int(frame[NUMERIC].isna().sum().sum()),
        "total_revenue": float(frame[TARGET].sum()),
        "average_revenue": float(frame[TARGET].mean()),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "seed": 42,
        "metrics": {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
            "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        },
        "regions": summary.to_dict(orient="records"),
        "categories": {field: sorted(frame[field].unique().tolist()) for field in CATEGORICAL},
        "holdout": [
            {"actual": float(actual), "predicted": float(predicted)}
            for actual, predicted in zip(y_test, predictions, strict=True)
        ],
    }
    return Analysis(model, report)


def predict(analysis, payload):
    if not isinstance(payload, dict) or set(payload) != set(FEATURES):
        raise ValueError("Provide exactly marketing_spend, visits, region, channel")
    for field in NUMERIC:
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number")
        if not np.isfinite(value) or not 0 <= value <= 1_000_000_000:
            raise ValueError(f"{field} must be finite and between 0 and 1000000000")
    for field in CATEGORICAL:
        if (
            not isinstance(payload[field], str)
            or payload[field] not in analysis.report["categories"][field]
        ):
            raise ValueError(f"{field} must be one of the analyzed categories")
    estimate = float(analysis.model.predict(pd.DataFrame([payload], columns=FEATURES))[0])
    return {"predicted_revenue": max(0.0, estimate)}
