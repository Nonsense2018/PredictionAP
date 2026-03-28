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


def test_fire_one_county_one_day_structure(monkeypatch) -> None:
    fire_module = load_module("scripts/fire/fetch_eonet_wildfire_data.py")

    payload = {
        "events": [
            {
                "id": "EONET_1",
                "title": "Test Wildfire",
                "geometry": [
                    {"date": "2024-08-01T00:00:00Z", "type": "Point", "coordinates": [-119.7, 36.8]}
                ],
            }
        ]
    }

    monkeypatch.setattr(fire_module.requests, "get", lambda *args, **kwargs: DummyResponse(payload))

    events = fire_module.fetch_wildfire_events(date(2024, 8, 1), date(2024, 8, 1))
    assert not events.empty

    centroids = pd.DataFrame(
        [
            {"county": "Fresno", "latitude": 36.7, "longitude": -119.8},
        ]
    )

    daily = fire_module.build_county_daily_fire(
        events=events,
        centroids=centroids,
        radius_km=150.0,
        start_date=date(2024, 8, 1),
        end_date=date(2024, 8, 1),
    )

    assert len(daily) == 1
    assert {"smoke_present", "min_fire_distance_km", "fire_event_count_radius"}.issubset(daily.columns)
    assert daily.loc[0, "smoke_present"] in (0, 1)
    assert pd.notna(daily.loc[0, "min_fire_distance_km"])
