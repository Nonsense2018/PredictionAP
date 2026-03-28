"""
Compute county centroids for San Joaquin Valley counties.

Inputs:
- configs/counties_sjv.json
- A California county shapefile somewhere under data/raw/geo/

Output:
- data/processed/geo/counties_centroids.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COUNTIES_CONFIG_PATH = PROJECT_ROOT / "configs" / "counties_sjv.json"
RAW_GEO_DIR = PROJECT_ROOT / "data" / "raw" / "geo"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"


def _normalize_county_name(name: str) -> str:
    """Normalize county names so config/shapefile matching is robust."""
    value = str(name).strip().lower()
    value = value.replace(" county", "")
    return value


def load_county_list(config_path: Path) -> list[str]:
    """Load county names from a JSON config file."""
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        county_list = payload
    elif isinstance(payload, dict) and "counties" in payload and isinstance(payload["counties"], list):
        county_list = payload["counties"]
    else:
        raise ValueError(
            "Unsupported counties JSON format. "
            "Use either a list of county names or {'counties': [...]}."
        )

    cleaned = [str(item).strip() for item in county_list if str(item).strip()]
    if not cleaned:
        raise ValueError("County list is empty in configs/counties_sjv.json")
    return cleaned


def find_county_shapefile(raw_geo_dir: Path) -> Path:
    """Find one shapefile to use for county geometry."""
    shapefiles = sorted(raw_geo_dir.rglob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No .shp files found under {raw_geo_dir}")
    if len(shapefiles) > 1:
        print(f"Found multiple shapefiles; using first: {shapefiles[0]}")
    return shapefiles[0]


def detect_county_name_column(gdf: gpd.GeoDataFrame) -> str:
    """Detect the most likely county-name column."""
    candidates = ["NAME", "name", "COUNTY", "county", "COUNTY_NAM", "NAMELSAD"]
    for candidate in candidates:
        if candidate in gdf.columns:
            return candidate
    raise ValueError(
        "Could not detect county name column in shapefile. "
        "Update the candidate column list in detect_county_name_column()."
    )


def build_centroids_dataframe(counties: list[str], county_gdf: gpd.GeoDataFrame, county_col: str) -> pd.DataFrame:
    """Filter to requested counties and compute WGS84 centroid points."""
    wanted = {_normalize_county_name(name): name for name in counties}
    county_gdf = county_gdf.copy()
    county_gdf["_county_norm"] = county_gdf[county_col].apply(_normalize_county_name)

    filtered = county_gdf[county_gdf["_county_norm"].isin(wanted.keys())].copy()
    if filtered.empty:
        raise ValueError("No requested counties matched shapefile county names.")

    # Some shapefiles can contain multiple geometries per county; dissolve first.
    filtered = filtered.dissolve(by="_county_norm", as_index=False)
    filtered["county"] = filtered["_county_norm"].map(wanted)

    # Compute centroids in a projected CRS to avoid geographic centroid warnings.
    projected = filtered.to_crs(epsg=3310)  # California Albers
    centroid_points = projected.geometry.centroid
    centroid_wgs84 = gpd.GeoSeries(centroid_points, crs="EPSG:3310").to_crs(epsg=4326)

    result = pd.DataFrame(
        {
            "county": filtered["county"].astype(str).str.strip(),
            "longitude": centroid_wgs84.x,
            "latitude": centroid_wgs84.y,
        }
    )
    result = result.sort_values("county").reset_index(drop=True)
    return result


def main() -> None:
    """Run centroid generation end-to-end."""
    if not COUNTIES_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {COUNTIES_CONFIG_PATH}")
    if not RAW_GEO_DIR.exists():
        raise FileNotFoundError(f"Missing raw geo directory: {RAW_GEO_DIR}")

    counties = load_county_list(COUNTIES_CONFIG_PATH)
    shapefile_path = find_county_shapefile(RAW_GEO_DIR)
    print(f"Using shapefile: {shapefile_path}")

    gdf = gpd.read_file(shapefile_path)
    county_col = detect_county_name_column(gdf)
    centroids_df = build_centroids_dataframe(counties, gdf, county_col)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    centroids_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Wrote centroids to: {OUTPUT_CSV_PATH}")
    print(f"Rows written: {len(centroids_df)}")


if __name__ == "__main__":
    main()
