"""
Check AirNow historical data availability for county centroids.

Input:
- data/processed/geo/counties_centroids.csv
- AIRNOW_API_KEY from .env

Output:
- results/logs/airnow_availability.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTROIDS_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"
OUTPUT_LOG_PATH = PROJECT_ROOT / "results" / "logs" / "airnow_availability.csv"
ENV_PATH = PROJECT_ROOT / ".env"

AIRNOW_ENDPOINT = "https://www.airnowapi.org/aq/data/"
TEST_DATE = "2020-09-10"
REQUEST_TIMEOUT_SECONDS = 30


def load_api_key() -> str:
    """Load AirNow API key from .env."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("AIRNOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError("AIRNOW_API_KEY is missing in .env")
    return api_key


def load_centroids(path: Path) -> pd.DataFrame:
    """Read centroids CSV and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing centroid file: {path}")

    df = pd.read_csv(path)
    required_columns = {"county", "latitude", "longitude"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Centroid file missing columns: {sorted(missing)}")
    return df


def build_airnow_params(api_key: str, latitude: float, longitude: float, date_str: str) -> dict[str, str]:
    """Construct query parameters for one centroid/date lookup."""
    bbox_half_size = 0.25
    min_lon = longitude - bbox_half_size
    min_lat = latitude - bbox_half_size
    max_lon = longitude + bbox_half_size
    max_lat = latitude + bbox_half_size

    return {
        "startDate": f"{date_str}T00",
        "endDate": f"{date_str}T23",
        "parameters": "PM25",
        "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "dataType": "B",
        "format": "application/json",
        "verbose": "0",
        "monitorType": "0",
        "includerawconcentrations": "0",
        "API_KEY": api_key,
    }


def check_one_county(api_key: str, county: str, latitude: float, longitude: float) -> dict[str, object]:
    """Call AirNow and return one result row."""
    params = build_airnow_params(api_key, latitude, longitude, TEST_DATE)

    try:
        response = requests.get(AIRNOW_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        payload = response.json()
        record_count = len(payload) if isinstance(payload, list) else 0
        status = "data_found" if record_count > 0 else "no_data"

        return {
            "county": county,
            "latitude": latitude,
            "longitude": longitude,
            "test_date": TEST_DATE,
            "status": status,
            "record_count": record_count,
            "error_message": "",
        }
    except requests.RequestException as exc:
        return {
            "county": county,
            "latitude": latitude,
            "longitude": longitude,
            "test_date": TEST_DATE,
            "status": "request_error",
            "record_count": 0,
            "error_message": str(exc),
        }
    except ValueError as exc:
        return {
            "county": county,
            "latitude": latitude,
            "longitude": longitude,
            "test_date": TEST_DATE,
            "status": "parse_error",
            "record_count": 0,
            "error_message": str(exc),
        }


def main() -> None:
    """Run AirNow availability checks for each county centroid."""
    api_key = load_api_key()
    centroids_df = load_centroids(CENTROIDS_CSV_PATH)

    results: list[dict[str, object]] = []
    for row in centroids_df.itertuples(index=False):
        county = str(row.county)
        latitude = float(row.latitude)
        longitude = float(row.longitude)

        print(f"Checking {county} ({latitude:.4f}, {longitude:.4f})")
        result = check_one_county(api_key, county, latitude, longitude)
        results.append(result)

    output_df = pd.DataFrame(results)
    OUTPUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_LOG_PATH, index=False)

    print(f"Wrote availability log: {OUTPUT_LOG_PATH}")
    print(output_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
