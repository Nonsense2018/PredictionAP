# sjv-aqi-risk

End-to-end research pipeline for San Joaquin Valley next-day AQI prediction and AQI exceedance risk.

## Pipeline steps

1. Generate county centroids from county shapefile
2. Collect AirNow historical air-quality data
3. Collect Open-Meteo historical weather data
4. Collect NASA EONET wildfire event indicators
5. Merge county-day datasets
6. Build lag/rolling features and next-day targets
7. Train/evaluate regression and classification models

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Copy `.env.example` to `.env`
3. Fill `AIRNOW_API_KEY` in `.env`
4. Place a California county shapefile (`.shp` and sidecar files) under `data/raw/geo/`

## Run

1. Run setup check:
   - `python3 scripts/00_setup_check.py`
2. Run full pipeline:
   - `python3 scripts/run_pipeline.py`

## Main outputs

- `data/processed/geo/counties_centroids.csv`
- `data/processed/air/county_air_daily.csv`
- `data/processed/met/county_met_daily.csv`
- `data/processed/fire/county_fire_daily.csv`
- `data/processed/modeling/merged_daily_county.csv`
- `data/processed/modeling/features_dataset.csv`
- `results/models/metrics.json`
- `results/models/metrics.csv`
