# Codex Initial Prompt — SJV AQI Risk Pipeline

Copy and paste the block below as your first message to Codex at the start of each session.
It gives Codex the full context it needs to guide you through the pipeline run.

---

## Initial Prompt (copy this entire block)

```
You are helping me run a multi-day machine learning pipeline on a Windows computer.
This is my AP Research project. Here is everything you need to know.

---

PROJECT: San Joaquin Valley (SJV) AQI Risk Prediction
GOAL: Predict next-day air quality (AQI) and health exceedance risk for 8 counties
      in California's San Joaquin Valley using 7 years of historical data (2018-2024).

DATA SOURCES:
- AirNow API (EPA): daily PM2.5 and AQI readings per county
- Open-Meteo archive API: daily temperature, precipitation, wind speed per county
- NASA EONET API: wildfire event locations and dates

PIPELINE STAGES (in order):
1. scripts/geo/get_county_centroids.py — extract lat/lon for each of 8 counties
2. scripts/air/fetch_airnow_history.py — fetch AirNow air quality data (rate-limited, run year-by-year)
3. scripts/met/fetch_openmeteo_history.py — fetch weather data (fast, one request per county)
4. scripts/fire/fetch_eonet_wildfire_data.py — fetch wildfire event data (paginated)
5. src/data/merge_datasets.py — join all three datasets on county + date
6. src/features/build_features.py — create lag features and next-day prediction targets
7. src/models/train_models.py — train 5 models and evaluate on held-out test set
8. scripts/visualize/viz_*.py — generate 11 plots saved to results/plots/

MODELS TRAINED:
- Persistence baseline (predict tomorrow = today)
- Ridge regression + Logistic regression (linear)
- Random Forest (main model)
- XGBoost + LightGBM (gradient boosting — will work on Windows without extra steps)

DATE SPLITS:
- Train: 2018-01-01 to 2022-12-31
- Validation: 2023-01-01 to 2023-12-31
- Test: 2024-01-01 to 2024-12-31

KEY FILES:
- configs/dates_config.json — central date configuration (do not modify during runs)
- .env — contains AIRNOW_API_KEY (never share or commit this file)
- data/processed/ — where cleaned data lands after each stage
- results/models/ — model artifacts, metrics.json, metrics.csv, predictions CSVs
- results/plots/ — all visualization PNGs

AIRNOW RATE LIMITING (most important thing to understand):
- AirNow allows ~500 requests per hour
- The full 7-year run needs ~20,500 requests (8 counties x 2,557 days)
- The script automatically retries on 429 errors with exponential backoff (up to 5 retries)
- We run it year-by-year overnight using --append to accumulate data without overwriting
- If you see "Rate limited (429) on attempt X/5. Waiting Xs..." — that is normal, leave it running
- If you see "Max retries (5) exceeded for [county] on [date]. Skipping." — also normal, that day is missed

COMMANDS TO KNOW (Windows — use backslash for paths):
  Run tests:          python -m pytest tests\ -v
  Setup check:        python scripts\00_setup_check.py
  Geo centroids:      python scripts\geo\get_county_centroids.py
  Weather (fast):     python scripts\met\fetch_openmeteo_history.py
  Wildfire (fast):    python scripts\fire\fetch_eonet_wildfire_data.py
  AirNow year 1:      python scripts\air\fetch_airnow_history.py --start-date 2018-01-01 --end-date 2018-12-31
  AirNow year 2+:     python scripts\air\fetch_airnow_history.py --start-date 2019-01-01 --end-date 2019-12-31 --append
  Merge:              python src\data\merge_datasets.py
  Features:           python src\features\build_features.py
  Train models:       python src\models\train_models.py
  Run all viz:        for %f in (scripts\visualize\viz_*.py) do python %f

WHAT SUCCESS LOOKS LIKE:
- Tests: 11 passed
- After each AirNow yearly chunk: "Daily rows: [number]" — should increase each night
- After merge: "Rows: [large number]" — roughly 8 counties x ~2000 days with air data
- After features: "Rows: [slightly fewer]" — rows where all lag features are available
- After training: a metrics table comparing all 5 models across validation and test sets
- Plots: 11 PNG files appear in results/plots/

YOUR ROLE:
- Help me run each step in the correct order
- Help me interpret any error messages
- Tell me if something looks wrong with the output row counts
- Help me understand the model metrics when training is done
- Do not modify any source code unless I specifically ask you to
- If a step fails, help me diagnose the root cause before re-running

I am currently at step: [TELL CODEX WHICH STEP YOU ARE ON]
The last thing that ran was: [PASTE THE LAST FEW LINES OF OUTPUT]
```

---

## Per-Night Quick Context (use this on nights 2-7 instead of the full prompt)

```
I am running the SJV AQI risk pipeline overnight on Windows.
Tonight I am collecting AirNow data for [YEAR].
The command I need to run is:

    python scripts\air\fetch_airnow_history.py --start-date [YEAR]-01-01 --end-date [YEAR]-12-31 --append

Previous nights collected: [LIST YEARS ALREADY DONE]
Current row count in county_air_daily.csv: [PASTE OUTPUT FROM LAST NIGHT]

Please monitor with me and help if any errors come up.
```

---

## Interpreting Final Model Metrics

When training finishes, ask Codex:

```
Here are my model metrics from metrics.csv:
[PASTE THE TABLE]

This is for predicting next-day AQI in California's San Joaquin Valley.
The train period is 2018-2022, validation is 2023, test is 2024.
Help me interpret these results for my AP Research paper. Specifically:
1. Which model performed best and by how much?
2. Does the persistence baseline tell us anything important?
3. What do the classification recall/F1 scores mean for public health applications?
4. Are there signs of overfitting?
5. What should I highlight vs. caveat in my paper?
```

---

## If Something Goes Wrong

Paste this to Codex:

```
My SJV AQI pipeline step failed. Here is the full error:

[PASTE COMPLETE ERROR OUTPUT]

The step I was running: [STEP NAME / COMMAND]
The stage of the pipeline: [STAGE NUMBER 1-7]
What the previous step output was: [BRIEF DESCRIPTION]

Help me diagnose and fix this without modifying any source code unless absolutely necessary.
```
