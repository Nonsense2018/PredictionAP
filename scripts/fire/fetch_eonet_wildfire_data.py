"""Fetch wildfire event data from NASA EONET and build county-day fire indicators."""

from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.date_config import get_data_collection_range, parse_iso_date


CENTROIDS_PATH = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"
RAW_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "fire" / "eonet_wildfire_events_raw.csv"
PROCESSED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "fire" / "county_fire_daily.csv"
ENV_PATH = PROJECT_ROOT / ".env"

EONET_EVENTS_ENDPOINT = "https://eonet.gsfc.nasa.gov/api/v3/events"
REQUEST_TIMEOUT_SECONDS = 60
EARTH_RADIUS_KM = 6371.0
DEFAULT_RADIUS_KM = 150.0


def iter_dates(start_date: date, end_date: date):
    """Yield each day in an inclusive date range."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    """Parse optional CLI date overrides."""
    parser = argparse.ArgumentParser(description="Fetch wildfire county-day indicators")
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


def load_radius_km() -> float:
    """Load distance radius used to assign events to county centroids."""
    load_dotenv(dotenv_path=ENV_PATH)
    value = os.getenv("FIRE_RADIUS_KM", "").strip()
    if not value:
        return DEFAULT_RADIUS_KM

    radius = float(value)
    if radius <= 0:
        raise ValueError("FIRE_RADIUS_KM must be positive.")
    return radius


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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def parse_geometry_point(geometry: dict) -> tuple[float, float] | None:
    """Extract one lon/lat point from EONET geometry payload."""
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        return None

    first = coordinates[0]
    if isinstance(first, (float, int)) and len(coordinates) >= 2:
        lon = float(coordinates[0])
        lat = float(coordinates[1])
        return lon, lat

    if isinstance(first, list):
        flat_points = []

        def collect_points(node: list) -> None:
            if len(node) >= 2 and isinstance(node[0], (float, int)) and isinstance(node[1], (float, int)):
                flat_points.append((float(node[0]), float(node[1])))
                return
            for item in node:
                if isinstance(item, list):
                    collect_points(item)

        collect_points(coordinates)
        if flat_points:
            lon = sum(point[0] for point in flat_points) / len(flat_points)
            lat = sum(point[1] for point in flat_points) / len(flat_points)
            return lon, lat

    return None


def fetch_wildfire_events(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch wildfire events from EONET for the selected window, paginating if needed."""
    PAGE_SIZE = 2000
    base_params = {
        "status": "all",
        "category": "wildfires",
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "limit": PAGE_SIZE,
    }

    all_rows: list[dict[str, object]] = []
    offset = 0

    while True:
        params = {**base_params, "offset": offset}
        response = requests.get(EONET_EVENTS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        payload = response.json()
        events = payload.get("events", [])

        for event in events:
            event_id = event.get("id")
            title = event.get("title")
            for geometry in event.get("geometry", []):
                point = parse_geometry_point(geometry)
                if point is None:
                    continue

                lon, lat = point
                date_value = pd.to_datetime(geometry.get("date"), errors="coerce")
                if pd.isna(date_value):
                    continue

                all_rows.append(
                    {
                        "event_id": event_id,
                        "event_title": title,
                        "date": date_value.strftime("%Y-%m-%d"),
                        "longitude": lon,
                        "latitude": lat,
                    }
                )

        # Stop when fewer events than page size were returned (last page)
        if len(events) < PAGE_SIZE:
            break
        offset += len(events)

    return pd.DataFrame(all_rows)


def build_county_daily_fire(events: pd.DataFrame, centroids: pd.DataFrame, radius_km: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Count nearby wildfire events for each county/day."""
    all_dates = [d.isoformat() for d in iter_dates(start_date, end_date)]
    rows: list[dict[str, object]] = []

    for centroid in centroids.itertuples(index=False):
        county = str(centroid.county)
        c_lat = float(centroid.latitude)
        c_lon = float(centroid.longitude)

        for day in all_dates:
            if events.empty:
                nearby_count = 0
                min_distance_km = pd.NA
            else:
                day_events = events[events["date"] == day]
                count = 0
                distances: list[float] = []
                for event in day_events.itertuples(index=False):
                    distance = haversine_km(c_lat, c_lon, float(event.latitude), float(event.longitude))
                    distances.append(distance)
                    if distance <= radius_km:
                        count += 1
                nearby_count = count
                min_distance_km = min(distances) if distances else pd.NA

            rows.append(
                {
                    "county": county,
                    "date": day,
                    "fire_event_count_radius": nearby_count,
                    "smoke_present": int(nearby_count > 0),
                    "min_fire_distance_km": min_distance_km,
                    "fire_radius_km": radius_km,
                }
            )

    return pd.DataFrame(rows).sort_values(["county", "date"]).reset_index(drop=True)


def main() -> None:
    """Run wildfire event collection and county/day feature creation."""
    args = parse_args()
    start_date, end_date = get_date_range(args.start_date, args.end_date)
    radius_km = load_radius_km()
    centroids = load_centroids()

    print(f"Collecting wildfire events from {start_date} to {end_date}")
    print(f"Using centroid assignment radius: {radius_km} km")

    try:
        wildfire_events = fetch_wildfire_events(start_date, end_date)
    except requests.RequestException as exc:
        print(f"Wildfire request failed: {exc}")
        wildfire_events = pd.DataFrame(columns=["event_id", "event_title", "date", "longitude", "latitude"])

    RAW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    wildfire_events.to_csv(RAW_OUTPUT_PATH, index=False)

    county_daily = build_county_daily_fire(wildfire_events, centroids, radius_km, start_date, end_date)
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    county_daily.to_csv(PROCESSED_OUTPUT_PATH, index=False)

    print(f"Wrote raw wildfire events: {RAW_OUTPUT_PATH}")
    print(f"Wrote county daily wildfire indicators: {PROCESSED_OUTPUT_PATH}")
    print(f"Daily rows: {len(county_daily)}")


if __name__ == "__main__":
    main()
