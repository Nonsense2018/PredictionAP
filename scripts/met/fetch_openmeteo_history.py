"""Fetch county-level historical meteorology from Open-Meteo."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.date_config import get_data_collection_range, parse_iso_date


CENTROIDS_PATH = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "met" / "openmeteo_daily_raw.csv"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "met" / "county_met_daily.csv"

OPEN_METEO_ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SECONDS = 60


def iter_dates(start_date: date, end_date: date):
    """Yield each day in an inclusive date range."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    """Parse optional CLI date overrides."""
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo county-day data")
    parser.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    return parser.parse_args()


def get_date_range(start_override: str | None, end_override: str | None) -> tuple[date, date]:
    """Resolve date range from CLI override or central config."""
    config_start, config_end = get_data_collection_range()
    start_date = parse_iso_date(start_override) if start_override else config_start
    end_date = parse_iso_date(end_override) if end_override else config_end

    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    return start_date, end_date


def load_centroids() -> pd.DataFrame:
    """Load county centroid coordinates."""
    if not CENTROIDS_PATH.exists():
        raise FileNotFoundError(f"Missing centroid file: {CENTROIDS_PATH}")

    centroids = pd.read_csv(CENTROIDS_PATH)
    required = {"county", "latitude", "longitude"}
    missing = required - set(centroids.columns)
    if missing:
        raise ValueError(f"Centroids file is missing columns: {sorted(missing)}")
    return centroids


def fetch_county_weather_for_day(county: str, latitude: float, longitude: float, day: date) -> pd.DataFrame:
    """Fetch one county's weather for one day."""
    day_str = day.isoformat()
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": day_str,
        "end_date": day_str,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    frame = pd.DataFrame(daily)
    frame["county"] = county
    frame["latitude"] = latitude
    frame["longitude"] = longitude

    if "temperature_2m_max" in frame.columns and "temperature_2m_min" in frame.columns:
        frame["temperature_2m_mean"] = (frame["temperature_2m_max"] + frame["temperature_2m_min"]) / 2.0

    frame = frame.rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame


def fetch_county_weather_range(county: str, latitude: float, longitude: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch one county's weather for a full date range in a single API call."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "UTC",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()

    frame = pd.DataFrame(daily)
    frame["county"] = county
    frame["latitude"] = latitude
    frame["longitude"] = longitude

    if "temperature_2m_max" in frame.columns and "temperature_2m_min" in frame.columns:
        frame["temperature_2m_mean"] = (frame["temperature_2m_max"] + frame["temperature_2m_min"]) / 2.0

    frame = frame.rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame


def build_processed_weather(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Select and clean columns for model-ready weather data."""
    if raw_frame.empty:
        return pd.DataFrame(
            columns=[
                "county",
                "date",
                "temperature_2m_mean",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]
        )

    weather = raw_frame.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    columns = [
        "county",
        "date",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
    for column in columns:
        if column not in weather.columns:
            weather[column] = pd.NA

    weather = weather[columns].drop_duplicates(subset=["county", "date"]).sort_values(["county", "date"]).reset_index(drop=True)
    return weather


def main() -> None:
    """Run Open-Meteo collection and save county-day files."""
    args = parse_args()
    start_date, end_date = get_date_range(args.start_date, args.end_date)
    centroids = load_centroids()

    print(f"Collecting Open-Meteo data from {start_date} to {end_date}")

    county_frames: list[pd.DataFrame] = []
    for row in centroids.itertuples(index=False):
        county = str(row.county)
        lat = float(row.latitude)
        lon = float(row.longitude)

        print(f"Fetching weather for {county}")
        try:
            county_frame = fetch_county_weather_range(county, lat, lon, start_date, end_date)
        except requests.RequestException as exc:
            print(f"Request failed for {county}: {exc}")
            county_frame = pd.DataFrame()

        if not county_frame.empty:
            county_frames.append(county_frame)

    raw_weather = pd.concat(county_frames, ignore_index=True) if county_frames else pd.DataFrame()
    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_weather.to_csv(RAW_OUTPUT_PATH, index=False)

    processed_weather = build_processed_weather(raw_weather)
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_weather.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print(f"Wrote raw weather data: {RAW_OUTPUT_PATH}")
    print(f"Wrote county daily weather data: {PROCESSED_OUTPUT_PATH}")
    print(f"Daily rows: {len(processed_weather)}")


if __name__ == "__main__":
    main()
