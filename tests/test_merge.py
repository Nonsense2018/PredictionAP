from __future__ import annotations

from pathlib import Path

import pandas as pd

from conftest import load_module


def test_merge_on_tiny_samples(tmp_path) -> None:
    merge_module = load_module("src/data/merge_datasets.py")
    fixtures = Path(__file__).resolve().parent / "fixtures"

    merge_module.AIR_PATH = fixtures / "air_sample.csv"
    merge_module.MET_PATH = fixtures / "weather_sample.csv"
    merge_module.FIRE_PATH = fixtures / "fire_sample.csv"
    merge_module.OUTPUT_PATH = tmp_path / "merged.csv"

    merge_module.main()

    merged = pd.read_csv(merge_module.OUTPUT_PATH)
    assert len(merged) == 3
    assert {"county", "date", "aqi_mean", "temperature_2m_mean", "fire_event_count_radius"}.issubset(merged.columns)

    dupes = merged.duplicated(subset=["county", "date"]).sum()
    assert dupes == 0

    fresno_day = merged[(merged["county"] == "Fresno") & (merged["date"] == "2024-01-02")]
    assert len(fresno_day) == 1
    assert float(fresno_day.iloc[0]["aqi_mean"]) == 120.0
