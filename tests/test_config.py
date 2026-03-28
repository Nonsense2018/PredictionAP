from __future__ import annotations

import json
from pathlib import Path


EXPECTED_COUNTIES = {
    "San Joaquin",
    "Stanislaus",
    "Merced",
    "Madera",
    "Fresno",
    "Kings",
    "Tulare",
    "Kern",
}


def test_counties_config_valid() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "counties_sjv.json"
    assert config_path.exists(), "counties_sjv.json should exist"

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "counties" in payload

    counties = payload["counties"]
    assert isinstance(counties, list)
    assert len(counties) == 8
    assert "Stanislaus" in counties
    assert set(counties) == EXPECTED_COUNTIES
    assert all(isinstance(name, str) for name in counties)
