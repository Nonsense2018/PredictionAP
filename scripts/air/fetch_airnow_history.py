"""Fetch historical AirNow data and build a county-day air quality dataset."""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import time

import pandas as pd
import requests
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.date_config import get_data_collection_range, parse_iso_date


CENTROIDS_PATH = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "air" / "airnow_records_raw.csv"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
ENV_PATH = PROJECT_ROOT / ".env"

AIRNOW_ENDPOINT = "https://www.airnowapi.org/aq/data/"
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
BASE_WAIT_SECONDS = 2


def iter_dates(start_date: date, end_date: date):
    """Yield each day in an inclusive date range."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    """Parse optional CLI date overrides."""
    parser = argparse.ArgumentParser(description="Fetch AirNow historical county-day data")
    parser.add_argument("--start-date", type=str, default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--append", action="store_true", help="Append to existing raw records instead of overwriting")
    parser.add_argument("--bbox", type=float, default=None,
                        help="Override bbox half-size in degrees for ALL counties (e.g. 0.5). "
                             "Use --county-bbox for per-county overrides.")
    parser.add_argument("--county-bbox", type=str, default=None,
                        help="Per-county bbox overrides as JSON, e.g. '{\"Kern\":0.5,\"Kings\":1.0,\"Madera\":0.5}'")
    return parser.parse_args()


def get_date_range(start_override: str | None, end_override: str | None) -> tuple[date, date]:
    """Resolve date range from CLI override or central config."""
    config_start, config_end = get_data_collection_range()

    start_date = parse_iso_date(start_override) if start_override else config_start
    end_date = parse_iso_date(end_override) if end_override else config_end

    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    return start_date, end_date


def load_api_key() -> str:
    """Load AIRNOW_API_KEY from .env."""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("AIRNOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError("AIRNOW_API_KEY is missing in .env")
    return api_key


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


def build_params(api_key: str, latitude: float, longitude: float, day: date,
                 bbox_half_size: float = 0.25) -> dict[str, str]:
    """Build AirNow request parameters for one county/day bbox."""
    min_lon = longitude - bbox_half_size
    min_lat = latitude - bbox_half_size
    max_lon = longitude + bbox_half_size
    max_lat = latitude + bbox_half_size
    day_str = day.isoformat()

    return {
        "startDate": f"{day_str}T00",
        "endDate": f"{day_str}T23",
        "parameters": "PM25",
        "BBOX": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "dataType": "B",
        "format": "application/json",
        "verbose": "1",
        "monitorType": "0",
        "includerawconcentrations": "1",
        "API_KEY": api_key,
    }


def fetch_county_records_for_day(
    api_key: str,
    county: str,
    latitude: float,
    longitude: float,
    day: date,
    bbox_half_size: float = 0.25,
) -> pd.DataFrame:
    """Fetch AirNow records for one county centroid bbox and one day.

    Retries up to MAX_RETRIES times on HTTP 429 (rate limit) with exponential backoff.
    All other HTTP errors are re-raised immediately.
    """
    params = build_params(api_key, latitude, longitude, day, bbox_half_size)
    wait = BASE_WAIT_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(AIRNOW_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                print(f"  Rate limited (429) on attempt {attempt}/{MAX_RETRIES}. Waiting {wait}s …")
                time.sleep(wait)
                wait = min(wait * 2, 60)
                continue
            raise

        payload = response.json()
        if not isinstance(payload, list):
            return pd.DataFrame()

        frame = pd.DataFrame(payload)
        if frame.empty:
            return frame

        frame["county"] = county
        frame["centroid_latitude"] = latitude
        frame["centroid_longitude"] = longitude
        frame["request_date"] = day.isoformat()
        return frame

    print(f"  Max retries ({MAX_RETRIES}) exceeded for {county} on {day.isoformat()}. Skipping.")
    return pd.DataFrame()


def aggregate_daily_air(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AirNow records into one row per county-day."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "county",
                "date",
                "aqi_mean",
                "aqi_max",
                "pm25_mean",
                "pm25_max",
                "observation_count",
            ]
        )

    records = frame.copy()
    if "DateObserved" in records.columns:
        date_series = pd.to_datetime(records["DateObserved"], errors="coerce")
    elif "request_date" in records.columns:
        date_series = pd.to_datetime(records["request_date"], errors="coerce")
    else:
        return pd.DataFrame(
            columns=[
                "county",
                "date",
                "aqi_mean",
                "aqi_max",
                "pm25_mean",
                "pm25_max",
                "observation_count",
            ]
        )

    records["date"] = date_series.dt.strftime("%Y-%m-%d")
    records["AQI"] = pd.to_numeric(records.get("AQI"), errors="coerce")
    records["RawConcentration"] = pd.to_numeric(records.get("RawConcentration"), errors="coerce")

    parameter_name = records.get("ParameterName", pd.Series("", index=records.index)).astype(str).str.upper()
    pm_mask = parameter_name.isin(["PM2.5", "PM25"])

    grouped = records.groupby(["county", "date"], dropna=False)
    output_rows: list[dict[str, object]] = []

    for (county, day), group in grouped:
        if pd.isna(day):
            continue

        aqi_values = group.loc[pm_mask.loc[group.index], "AQI"].dropna()
        if aqi_values.empty:
            aqi_values = group["AQI"].dropna()

        pm_values = group.loc[pm_mask.loc[group.index], "RawConcentration"].dropna()

        output_rows.append(
            {
                "county": county,
                "date": day,
                "aqi_mean": float(aqi_values.mean()) if not aqi_values.empty else pd.NA,
                "aqi_max": float(aqi_values.max()) if not aqi_values.empty else pd.NA,
                "pm25_mean": float(pm_values.mean()) if not pm_values.empty else pd.NA,
                "pm25_max": float(pm_values.max()) if not pm_values.empty else pd.NA,
                "observation_count": int(len(group)),
            }
        )

    return pd.DataFrame(output_rows).sort_values(["county", "date"]).reset_index(drop=True)


def main() -> None:
    """Run historical AirNow collection and aggregation."""
    import json as _json
    args = parse_args()
    api_key = load_api_key()
    start_date, end_date = get_date_range(args.start_date, args.end_date)
    centroids = load_centroids()

    # Build per-county bbox lookup
    county_bbox_override: dict[str, float] = {}
    if args.county_bbox:
        county_bbox_override = _json.loads(args.county_bbox)
    default_bbox = args.bbox if args.bbox else 0.25

    print(f"Collecting AirNow data from {start_date} to {end_date}")
    print(f"Default bbox half-size: {default_bbox}°")
    if county_bbox_override:
        print(f"Per-county bbox overrides: {county_bbox_override}")

    county_frames: list[pd.DataFrame] = []
    for row in centroids.itertuples(index=False):
        county = str(row.county)
        lat = float(row.latitude)
        lon = float(row.longitude)
        bbox = county_bbox_override.get(county, default_bbox)

        print(f"Fetching AirNow for {county} (bbox ±{bbox}°)")
        for day in iter_dates(start_date, end_date):
            try:
                county_frame = fetch_county_records_for_day(api_key, county, lat, lon, day, bbox)
            except requests.RequestException as exc:
                print(f"Request failed for {county} on {day.isoformat()}: {exc}")
                county_frame = pd.DataFrame()

            if not county_frame.empty:
                county_frames.append(county_frame)

    new_records = pd.concat(county_frames, ignore_index=True) if county_frames else pd.DataFrame()

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.append and RAW_OUTPUT_PATH.exists() and RAW_OUTPUT_PATH.stat().st_size > 1:
        existing = pd.read_csv(RAW_OUTPUT_PATH)
        raw_records = pd.concat([existing, new_records], ignore_index=True)
        print(f"Appended {len(new_records)} new records to {len(existing)} existing records.")
    else:
        raw_records = new_records

    raw_records.to_csv(RAW_OUTPUT_PATH, index=False)

    daily_air = aggregate_daily_air(raw_records)
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily_air.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print(f"Wrote raw AirNow records: {RAW_OUTPUT_PATH}")
    print(f"Wrote county daily air data: {PROCESSED_OUTPUT_PATH}")
    print(f"Daily rows: {len(daily_air)}")


if __name__ == "__main__":
    main()
