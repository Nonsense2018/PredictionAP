from __future__ import annotations

from datetime import date

from conftest import load_module


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_air_api_key_loads(monkeypatch) -> None:
    air_module = load_module("scripts/air/fetch_airnow_history.py")
    monkeypatch.setenv("AIRNOW_API_KEY", "unit-test-key")
    assert air_module.load_api_key() == "unit-test-key"


def test_air_one_county_one_day_pull_parseable(monkeypatch) -> None:
    air_module = load_module("scripts/air/fetch_airnow_history.py")

    sample_payload = [
        {
            "DateObserved": "2024-07-15",
            "HourObserved": 12,
            "ParameterName": "PM2.5",
            "AQI": 83,
            "RawConcentration": 19.2,
        }
    ]

    monkeypatch.setattr(air_module.requests, "get", lambda *args, **kwargs: DummyResponse(sample_payload))

    frame = air_module.fetch_county_records_for_day(
        api_key="unit-test-key",
        county="Fresno",
        latitude=36.7,
        longitude=-119.8,
        day=date(2024, 7, 15),
    )

    assert not frame.empty
    assert {"DateObserved", "AQI", "ParameterName", "county"}.issubset(frame.columns)


def test_air_one_county_one_day_request_success_even_if_no_records(monkeypatch) -> None:
    air_module = load_module("scripts/air/fetch_airnow_history.py")
    monkeypatch.setattr(air_module.requests, "get", lambda *args, **kwargs: DummyResponse([]))

    frame = air_module.fetch_county_records_for_day(
        api_key="unit-test-key",
        county="Kern",
        latitude=35.4,
        longitude=-119.2,
        day=date(2024, 7, 15),
    )

    assert frame.empty
