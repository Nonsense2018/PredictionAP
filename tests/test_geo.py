from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from conftest import load_module


def test_geo_centroid_pipeline_runs_and_writes_expected_output(tmp_path) -> None:
    geo_module = load_module("scripts/geo/get_county_centroids.py")

    counties = [
        "San Joaquin",
        "Stanislaus",
        "Merced",
        "Madera",
        "Fresno",
        "Kings",
        "Tulare",
        "Kern",
    ]

    config_path = tmp_path / "counties_sjv.json"
    config_path.write_text(json.dumps({"counties": counties}), encoding="utf-8")

    raw_geo_dir = tmp_path / "raw_geo"
    raw_geo_dir.mkdir()
    shapefile_path = raw_geo_dir / "ca_counties.shp"
    shapefile_path.write_text("placeholder", encoding="utf-8")

    output_path = tmp_path / "counties_centroids.csv"

    geometries = []
    for i in range(len(counties)):
        lon = -121.5 + (i * 0.35)
        lat = 35.0 + (i * 0.35)
        geometries.append(box(lon, lat, lon + 0.1, lat + 0.1))

    gdf = gpd.GeoDataFrame({"NAME": counties, "geometry": geometries}, crs="EPSG:4326")

    geo_module.COUNTIES_CONFIG_PATH = config_path
    geo_module.RAW_GEO_DIR = raw_geo_dir
    geo_module.OUTPUT_CSV_PATH = output_path

    def fake_find_shapefile(_: Path) -> Path:
        return shapefile_path

    geo_module.find_county_shapefile = fake_find_shapefile
    geo_module.gpd.read_file = lambda _path: gdf

    geo_module.main()

    assert output_path.exists()
    result = pd.read_csv(output_path)

    assert list(result.columns) == ["county", "longitude", "latitude"]
    assert len(result) == 8
    assert result["latitude"].between(34, 39).all()
    assert result["longitude"].between(-122, -117).all()
