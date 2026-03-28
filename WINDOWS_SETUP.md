# Windows Setup Guide — SJV AQI Risk Pipeline

Complete checklist for getting the pipeline running on a fresh Windows machine.

---

## 1. Python

Install **Python 3.10 or 3.11** (recommended) from https://www.python.org/downloads/

During installation:
- ✅ Check **"Add Python to PATH"**
- ✅ Check **"Install pip"**

Verify in Command Prompt:
```cmd
python --version
pip --version
```

> **Note:** On Windows the command is `python`, not `python3`. All `python3` references in these docs become `python` on Windows.

---

## 2. Clone the Repository

```cmd
git clone https://github.com/Nonsense2018/PredictionAP.git
cd PredictionAP
```

---

## 3. Install Dependencies

### Option A — Conda (Recommended, handles geopandas automatically)

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then:

```cmd
conda create -n sjv-aqi python=3.11
conda activate sjv-aqi
conda install -c conda-forge geopandas
pip install requests python-dotenv scikit-learn joblib pytest matplotlib seaborn xgboost lightgbm
```

### Option B — pip only (if you don't want Conda)

Geopandas requires GDAL and Fiona which are hard to install via pip on Windows.
Use the unofficial pre-built wheels from Christoph Gohlke:

```cmd
pip install pipwin
pipwin install gdal
pipwin install fiona
pipwin install shapely
pip install geopandas
pip install -r requirements.txt
```

> If pipwin fails, download the `.whl` files manually from:
> https://github.com/cgohlke/geospatial-wheels/releases
> Install in this order: GDAL → Fiona → Shapely → geopandas

---

## 4. Create Required Directories

These folders are gitignored and won't exist after cloning. Create them manually:

```cmd
mkdir data\raw\geo
mkdir data\raw\air
mkdir data\raw\met
mkdir data\raw\fire
mkdir results\logs
```

---

## 5. Download the County Shapefile

The US Census county shapefile is required for Step 1 (centroid generation).
It is **not included in the repo** (too large).

Download from:
```
https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip
```

Extract the zip. You should get these files:
```
tl_2023_us_county.shp
tl_2023_us_county.dbf
tl_2023_us_county.shx
tl_2023_us_county.prj
tl_2023_us_county.cpg
```

Copy all of them into:
```
PredictionAP\data\raw\geo\
```

---

## 6. Set Up Your API Key

Create a file called `.env` in the root of the project (same folder as `requirements.txt`):

```
AIRNOW_API_KEY=your_actual_key_here
FIRE_RADIUS_KM=150
```

Replace `your_actual_key_here` with the real AirNow API key.

> ⚠️ Never commit this file. It is already listed in `.gitignore`.

---

## 7. Verify the Setup

```cmd
python scripts\00_setup_check.py
```

Expected output: `Setup check passed.`

Then run the test suite:

```cmd
python -m pytest tests\ -v
```

Expected: `11 passed`.

If any test fails, do not proceed — fix the issue first.

---

## 8. Run the Pipeline (Overnight Strategy)

### Step 1 — Generate county centroids (run once, takes seconds)
```cmd
python scripts\geo\get_county_centroids.py
```

### Step 2 — Weather and wildfire data (runs fast, no rate limits)
```cmd
python scripts\met\fetch_openmeteo_history.py
python scripts\fire\fetch_eonet_wildfire_data.py
```

### Step 3 — AirNow data (the slow part — run year-by-year overnight)

AirNow has a rate limit of ~500 requests/hour. The full 7-year range
has ~20,500 requests. Run in yearly chunks across multiple nights:

**Night 1:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2018-01-01 --end-date 2018-12-31
```

**Night 2:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2019-01-01 --end-date 2019-12-31 --append
```

**Night 3:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2020-01-01 --end-date 2020-12-31 --append
```

**Night 4:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2021-01-01 --end-date 2021-12-31 --append
```

**Night 5:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2022-01-01 --end-date 2022-12-31 --append
```

**Night 6:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2023-01-01 --end-date 2023-12-31 --append
```

**Night 7:**
```cmd
python scripts\air\fetch_airnow_history.py --start-date 2024-01-01 --end-date 2024-12-31 --append
```

> **Important:** Always use `--append` after the first night. Without it, the previous data is overwritten.

### Step 4 — Merge, features, and models (after all air data is collected)
```cmd
python src\data\merge_datasets.py
python src\features\build_features.py
python src\models\train_models.py
```

### Step 5 — Generate visualizations
```cmd
python scripts\visualize\viz_aqi_timeseries.py
python scripts\visualize\viz_county_heatmap.py
python scripts\visualize\viz_exceedance_rate.py
python scripts\visualize\viz_wildfire_scatter.py
python scripts\visualize\viz_correlation_matrix.py
python scripts\visualize\viz_predicted_vs_actual.py
python scripts\visualize\viz_residuals.py
python scripts\visualize\viz_roc_curve.py
python scripts\visualize\viz_feature_importance.py
python scripts\visualize\viz_confusion_matrix.py
python scripts\visualize\viz_error_calendar.py
```

All plots are saved to `results\plots\`.

---

## 9. What to Watch For

| Warning | Meaning | Action |
|---------|---------|--------|
| `Rate limited (429) on attempt X/5` | AirNow rate limit hit | Script retries automatically — leave it running |
| `Max retries (5) exceeded for ... Skipping.` | A specific county/day was permanently skipped | Expected under heavy rate limiting — those days will be missing |
| `Request failed for ... : ...` | Network error | Check internet connection; re-run that year chunk |
| `One or more time splits are empty` | Not enough date range in collected data | Make sure all 7 years of air data are collected before training |

---

## 10. Preventing Windows Sleep

To stop Windows from sleeping during an overnight run:

```cmd
powercfg /change standby-timeout-ac 0
```

To restore after the run:

```cmd
powercfg /change standby-timeout-ac 30
```

Or set a custom power plan in Settings → Power & Sleep → Sleep → **Never**.

---

## Summary Checklist

- [ ] Python 3.10 or 3.11 installed and on PATH
- [ ] Repository cloned
- [ ] Dependencies installed (geopandas via conda-forge recommended)
- [ ] `data\raw\geo\` directory created and shapefile copied in
- [ ] `data\raw\air\`, `data\raw\met\`, `data\raw\fire\`, `results\logs\` created
- [ ] `.env` file created with `AIRNOW_API_KEY`
- [ ] `python scripts\00_setup_check.py` → passes
- [ ] `python -m pytest tests\ -v` → 11 passed
- [ ] Windows sleep disabled for overnight run
