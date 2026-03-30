# Windows Machine — Task Brief

## What Was Wrong

The initial AirNow data collection run completed successfully for 5 of the 8 SJV counties.
Three counties — **Kern, Kings, and Madera** — returned zero data for every day across all 7 years.

A live API diagnostic test confirmed the issue: those counties have AirNow monitoring stations,
but the stations are located farther from the county centroid than the default ±0.25° bounding box.
The fetch script was querying a box that was simply too small to capture any stations.

Diagnostic results for a test date (2022-08-15):

| County | ±0.25° (old) | ±0.50° | ±1.00° |
|--------|-------------|--------|--------|
| Kern   | 0 records   | 48     | 288    |
| Kings  | 0 records   | 0      | 48     |
| Madera | 0 records   | 96     | 336    |

## What Was Fixed

`scripts/air/fetch_airnow_history.py` now accepts two new CLI arguments:

- `--bbox FLOAT` — override the bbox half-size in degrees for all counties
- `--county-bbox JSON` — per-county bbox overrides as a JSON string

The fetch function now uses the per-county bbox when making API requests.
No other logic changed.

---

## What You Need to Do

### Step 1 — Pull the latest code

```
git pull origin main
```

### Step 2 — Re-fetch Kern, Kings, and Madera (run each one, one at a time)

Each command fetches the full 2018–2024 range for that county only and appends to the existing data.
These will be slow (~2557 requests per county). Run overnight.

**Kern** (needs ±0.5°):
```
python scripts\air\fetch_airnow_history.py --start-date 2018-01-01 --end-date 2024-12-31 --append --county-bbox "{\"Kern\":0.5}"
```

**Madera** (needs ±0.5°):
```
python scripts\air\fetch_airnow_history.py --start-date 2018-01-01 --end-date 2024-12-31 --append --county-bbox "{\"Madera\":0.5}"
```

**Kings** (needs ±1.0°):
```
python scripts\air\fetch_airnow_history.py --start-date 2018-01-01 --end-date 2024-12-31 --append --county-bbox "{\"Kings\":1.0}"
```

> Each command only fetches the county specified in `--county-bbox`.
> All other counties in the centroids file will use the default ±0.25° and return data
> that duplicates what you already have — the `--append` flag will merge it cleanly.
> Alternatively, if you want to skip already-collected counties entirely, temporarily
> remove the other 5 counties from `data/processed/geo/counties_centroids.csv` before
> running, then restore the file after.

### Step 3 — Re-run the pipeline after all 3 counties are collected

```
python src\data\merge_datasets.py
python src\features\build_features.py
python src\models\train_models.py
python scripts\visualize\run_all_visualizations.py
```

### Step 4 — Run diagnostics to confirm all 8 counties present

```
python scripts\diagnostics.py
```

Look for `[OK] All 8 counties present in raw data.` in section 2 and 3.

### Step 5 — Push results to GitHub

```
git add data\processed\ results\
git commit -m "Add Kern, Kings, Madera with wider bbox; retrain models"
git push origin main
```

---

## Why the Wider Bbox Is Justified (for the paper)

Rural counties in California's southern SJV have fewer EPA monitoring stations.
The county centroid (geographic center) may be far from any urban monitoring site.
Using a wider bounding box for these three counties is documented and methodologically
justified: it captures the nearest available station rather than returning no data.

Paper language:
> "For Kern, Kings, and Madera counties, the bounding box was expanded to 0.5°–1.0°
> to capture monitoring stations located farther from the county centroid, reflecting
> lower rural monitoring density in the southern San Joaquin Valley."
