from __future__ import annotations

import pandas as pd

from conftest import load_module


def test_feature_engineering_on_toy_dataframe(tmp_path) -> None:
    feature_module = load_module("src/features/build_features.py")

    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    aqi_values = [80, 100, 160, 120, 140, 130, 110, 90, 100, 80]

    merged = pd.DataFrame(
        {
            "county": ["Fresno"] * 10,
            "date": dates.strftime("%Y-%m-%d"),
            "aqi_mean": aqi_values,
            "temperature_2m_mean": [15] * 10,
            "fire_event_count_radius": [0, 1, 2, 1, 0, 0, 1, 1, 0, 0],
        }
    )

    merged_path = tmp_path / "merged.csv"
    output_path = tmp_path / "features.csv"
    merged.to_csv(merged_path, index=False)

    feature_module.MERGED_PATH = merged_path
    feature_module.OUTPUT_PATH = output_path

    feature_module.main()

    features = pd.read_csv(output_path)
    assert not features.empty

    row = features[features["date"] == "2024-01-08"]
    assert len(row) == 1

    assert float(row.iloc[0]["aqi_lag_1"]) == 110.0
    assert round(float(row.iloc[0]["aqi_roll3_mean"]), 2) == round((110 + 130 + 140) / 3.0, 2)
    assert float(row.iloc[0]["target_next_day_aqi"]) == 100.0

    labels = features["target_next_day_exceedance"].dropna().astype(int)
    assert set(labels.unique()).issubset({0, 1})
