from __future__ import annotations

import json

import numpy as np
import pandas as pd

from conftest import load_module


def test_models_train_predict_and_evaluate_on_tiny_data(tmp_path) -> None:
    model_module = load_module("src/models/train_models.py")

    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2018-01-01", end="2024-12-31", freq="30D")
    n = len(dates)

    feature_df = pd.DataFrame(
        {
            "county": ["Fresno"] * n,
            "date": dates.strftime("%Y-%m-%d"),
            "aqi_lag_1": rng.normal(100, 10, n),
            "aqi_lag_2": rng.normal(100, 12, n),
            "aqi_lag_3": rng.normal(98, 11, n),
            "aqi_roll3_mean": rng.normal(102, 8, n),
            "aqi_roll7_mean": rng.normal(101, 7, n),
            "temperature_2m_mean": rng.normal(20, 3, n),
            "fire_event_count_radius": rng.integers(0, 3, n),
            "target_next_day_aqi": rng.normal(110, 15, n),
            "target_next_day_exceedance": [0, 1] * (n // 2) + ([0] if n % 2 else []),
        }
    )

    features_path = tmp_path / "features.csv"
    model_dir = tmp_path / "model_outputs"
    feature_df.to_csv(features_path, index=False)

    model_module.FEATURES_PATH = features_path
    model_module.MODELS_DIR = model_dir

    model_module.main()

    assert (model_dir / "regression_predictions.csv").exists()
    assert (model_dir / "classification_predictions.csv").exists()
    assert (model_dir / "metrics.csv").exists()
    assert (model_dir / "metrics.json").exists()

    reg_preds = pd.read_csv(model_dir / "regression_predictions.csv")
    cls_preds = pd.read_csv(model_dir / "classification_predictions.csv")
    assert len(reg_preds) == len(cls_preds)
    assert len(reg_preds) > 0

    metrics_payload = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "validation" in metrics_payload
    assert "test" in metrics_payload
