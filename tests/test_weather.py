from __future__ import annotations

from datetime import date

import pandas as pd

from conftest import load_module


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_weather_one_county_short_range(monkeypatch) -> None:
    weather_module = load_module("scripts/met/fetch_openmeteo_history.py")

    def fake_get(*args, **kwargs):
        day = kwargs["params"]["start_date"]
        payload = {
            "daily": {
                "time": [day],
                "temperature_2m_max": [39.0],
                "temperature_2m_min": [20.0],
                "precipitation_sum": [0.0],
                "wind_speed_10m_max": [18.0],
            }
        }
        return DummyResponse(payload)

    monkeypatch.setattr(weather_module.requests, "get", fake_get)

    raw_frames = []
    for day in [date(2024, 7, 1), date(2024, 7, 2)]:
        raw_frames.append(
            weather_module.fetch_county_weather_for_day(
                county="Fresno",
                latitude=36.7,
                longitude=-119.8,
                day=day,
            )
        )

    raw = pd.concat(raw_frames, ignore_index=True)
    assert len(raw) == 2

    processed = weather_module.build_processed_weather(raw)
    assert len(processed) == 2

    expected_cols = {
        "county",
        "date",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    }
    assert expected_cols.issubset(processed.columns)

    parsed_dates = pd.to_datetime(processed["date"], errors="coerce")
    assert parsed_dates.notna().all()
