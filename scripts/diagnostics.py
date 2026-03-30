"""
SJV AQI Pipeline Diagnostics
Checks data completeness, AirNow coverage per county, and missing county causes.

Usage:
    python scripts/diagnostics.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from datetime import date

import pandas as pd
import requests
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

CENTROIDS_PATH   = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"
RAW_AIR_PATH     = PROJECT_ROOT / "data" / "raw" / "air" / "airnow_records_raw.csv"
PROC_AIR_PATH    = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
PROC_MET_PATH    = PROJECT_ROOT / "data" / "processed" / "met" / "county_met_daily.csv"
PROC_FIRE_PATH   = PROJECT_ROOT / "data" / "processed" / "fire" / "county_fire_daily.csv"
FEATURES_PATH    = PROJECT_ROOT / "data" / "processed" / "modeling" / "features_dataset.csv"
COUNTIES_CONFIG  = PROJECT_ROOT / "configs" / "counties_sjv.json"
LOG_PATH         = PROJECT_ROOT / "results" / "logs" / "airnow_queue.log"

EXPECTED_COUNTIES = ["Fresno", "Kern", "Kings", "Madera", "Merced",
                     "San Joaquin", "Stanislaus", "Tulare"]
STUDY_START = date(2018, 1, 1)
STUDY_END   = date(2024, 12, 31)
TOTAL_DAYS  = (STUDY_END - STUDY_START).days + 1

AIRNOW_ENDPOINT = "https://www.airnowapi.org/aq/data/"
BBOX_SIZES = [0.25, 0.5, 1.0]   # degrees, tested in order

SEP  = "=" * 62
SEP2 = "-" * 62

def hdr(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):  print(f"  [OK]   {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def err(msg):  print(f"  [FAIL] {msg}")
def info(msg): print(f"         {msg}")


# ── 1. Queue log status ────────────────────────────────────────────────────────

def check_queue_log():
    hdr("1. AirNow Queue Runner Log")
    if not LOG_PATH.exists():
        warn("No queue log found — queue runner may not have been started.")
        return

    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    info(f"Log lines: {len(lines)}")

    if any("All AirNow years complete" in l for l in lines):
        ok("Queue runner completed all years.")
    elif any("2018 data confirmed present" in l for l in lines):
        warn("2018 confirmed but queue may not have finished all years.")
    else:
        err("Queue runner appears stuck waiting for 2018 data — never moved to 2019+.")
        info("The 'year_is_done(2018)' check polls the PROCESSED file.")
        info("If fetch_airnow_history.py wrote raw but didn't process, it will loop forever.")
        info("ACTION NEEDED: Run each year manually with --append (see prompts below).")

    years_mentioned = set()
    for l in lines:
        for y in range(2018, 2025):
            if str(y) in l and "Starting AirNow fetch" in l:
                years_mentioned.add(y)
    if years_mentioned:
        ok(f"Years that started fetching: {sorted(years_mentioned)}")
    else:
        warn("No year fetch start lines found — only 2018 was attempted.")


# ── 2. Raw AirNow data ─────────────────────────────────────────────────────────

def check_raw_air():
    hdr("2. Raw AirNow Records")
    if not RAW_AIR_PATH.exists():
        err(f"Missing: {RAW_AIR_PATH}")
        return

    df = pd.read_csv(RAW_AIR_PATH)
    info(f"Total raw rows: {len(df):,}")
    info(f"Counties present: {sorted(df['county'].unique())}")

    if "request_date" in df.columns:
        df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce")
        yr_counts = df["request_date"].dt.year.value_counts().sort_index()
        info("Rows per year:")
        for yr, cnt in yr_counts.items():
            flag = "" if cnt > 500 else "  ← LOW"
            info(f"  {yr}: {cnt:,} rows{flag}")
    else:
        warn("No request_date column — cannot check year coverage.")

    missing = [c for c in EXPECTED_COUNTIES if c not in df["county"].unique()]
    if missing:
        err(f"Counties with ZERO raw rows: {missing}")
    else:
        ok("All 8 counties present in raw data.")


# ── 3. Processed air data ──────────────────────────────────────────────────────

def check_processed_air():
    hdr("3. Processed Air Quality Data (county_air_daily.csv)")
    if not PROC_AIR_PATH.exists():
        err(f"Missing: {PROC_AIR_PATH}")
        return

    df = pd.read_csv(PROC_AIR_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    info(f"Total rows: {len(df):,}")
    info(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    info(f"Counties present: {sorted(df['county'].unique())}")

    print()
    info("Coverage per county (rows / expected ~2557 days):")
    for county in EXPECTED_COUNTIES:
        cdf = df[df["county"] == county]
        if cdf.empty:
            err(f"  {county:<14} — NO DATA")
        else:
            pct = len(cdf) / TOTAL_DAYS * 100
            flag = "OK" if pct > 70 else "LOW"
            yr_range = f"{cdf['date'].min().year}–{cdf['date'].max().year}"
            info(f"  {county:<14}  {len(cdf):>5} rows  ({pct:.0f}% of study period)  [{yr_range}]  [{flag}]")

    print()
    info("Rows per year (all counties combined):")
    yr_counts = df["date"].dt.year.value_counts().sort_index()
    for yr, cnt in yr_counts.items():
        flag = "" if cnt > 1000 else "  ← LOW"
        info(f"  {yr}: {cnt:,}{flag}")


# ── 4. Met & fire data ─────────────────────────────────────────────────────────

def check_other_datasets():
    hdr("4. Meteorology & Wildfire Data")
    for label, path in [("Met", PROC_MET_PATH), ("Fire", PROC_FIRE_PATH)]:
        if not path.exists():
            err(f"{label}: MISSING — {path}")
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        counties = sorted(df["county"].unique())
        missing = [c for c in EXPECTED_COUNTIES if c not in counties]
        info(f"{label}: {len(df):,} rows | {df['date'].min().date()} → {df['date'].max().date()}")
        info(f"  Counties: {counties}")
        if missing:
            warn(f"  Missing counties: {missing}")
        else:
            ok(f"  All 8 counties present.")


# ── 5. Features dataset ────────────────────────────────────────────────────────

def check_features():
    hdr("5. Features Dataset (used for training)")
    if not FEATURES_PATH.exists():
        err(f"Missing: {FEATURES_PATH}")
        return

    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    info(f"Total rows: {len(df):,}")
    info(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    info(f"Counties: {sorted(df['county'].unique())}")
    info(f"Columns: {df.columns.tolist()}")

    print()
    for split_name, y1, y2 in [("Train", 2018, 2022), ("Val", 2023, 2023), ("Test", 2024, 2024)]:
        sdf = df[df["date"].dt.year.between(y1, y2)]
        exc = (sdf["target_next_day_exceedance"] == 1).sum() if "target_next_day_exceedance" in sdf else 0
        pct_exc = exc / len(sdf) * 100 if len(sdf) > 0 else 0
        info(f"  {split_name} ({y1}–{y2}): {len(sdf):,} rows | Exceedance days: {exc} ({pct_exc:.1f}%)")


# ── 6. Live AirNow bbox test ───────────────────────────────────────────────────

def test_airnow_bbox(api_key: str, centroids: pd.DataFrame):
    hdr("6. Live AirNow API — Bbox Coverage Test (2022-08-15)")
    test_date = "2022-08-15"   # peak wildfire season — best chance of data

    print()
    info(f"Testing bbox sizes {BBOX_SIZES}° for each county on {test_date}:")
    info(f"{'County':<14} {'0.25°':>8} {'0.50°':>8} {'1.00°':>8}  Notes")
    info(SEP2)

    results = {}
    for _, row in centroids.iterrows():
        county = row["county"]
        lat, lon = row["latitude"], row["longitude"]
        counts = []
        for bbox in BBOX_SIZES:
            params = {
                "startDate":              f"{test_date}T00",
                "endDate":                f"{test_date}T23",
                "parameters":            "PM25",
                "BBOX":                  f"{lon-bbox},{lat-bbox},{lon+bbox},{lat+bbox}",
                "dataType":              "B",
                "format":                "application/json",
                "verbose":               "1",
                "monitorType":           "0",
                "includerawconcentrations": "1",
                "API_KEY":               api_key,
            }
            try:
                r = requests.get(AIRNOW_ENDPOINT, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                n = len(data) if isinstance(data, list) else 0
            except Exception as e:
                n = -1
            counts.append(n)

        notes = ""
        if counts[0] == 0 and counts[1] == 0 and counts[2] == 0:
            notes = "NO STATIONS — genuinely uncovered"
        elif counts[0] == 0 and counts[1] > 0:
            notes = f"Needs bbox ≥ 0.5° to find stations"
        elif counts[0] == 0 and counts[2] > 0:
            notes = f"Needs bbox ≥ 1.0° to find stations"
        elif counts[0] > 0:
            notes = "OK at default bbox"

        c0 = str(counts[0]) if counts[0] >= 0 else "ERR"
        c1 = str(counts[1]) if counts[1] >= 0 else "ERR"
        c2 = str(counts[2]) if counts[2] >= 0 else "ERR"
        info(f"  {county:<14} {c0:>6}   {c1:>6}   {c2:>6}   {notes}")
        results[county] = counts

    return results


# ── 7. Missing county fix prompts ──────────────────────────────────────────────

def print_fix_prompts(bbox_results: dict | None):
    hdr("7. What To Do Next")

    missing = [c for c in EXPECTED_COUNTIES
               if PROC_AIR_PATH.exists() and c not in
               pd.read_csv(PROC_AIR_PATH)["county"].unique()]

    if not missing:
        ok("All 8 counties have air data — no fix needed for coverage.")
    else:
        err(f"Missing counties: {missing}")
        print()
        info("Option A — Widen the bounding box for missing counties only")
        info("  Add a per-county bbox override in fetch_airnow_history.py")
        info("  based on the bbox test results above, then re-run with --append.")
        print()
        info("Option B — Accept the 5-county dataset as-is")
        info("  Document as a limitation: Kern, Kings, Madera had no AirNow")
        info("  monitoring stations within ±0.25° of county centroid coordinates.")
        info("  This is common in rural CA counties.")

    print()
    info("Queue runner fix (for Windows machine):")
    info("  The queue runner got stuck because year_is_done(2018) checks the")
    info("  PROCESSED file, but the 2018 fetch may have written raw data without")
    info("  triggering the processing step. Run years manually instead:")
    print()
    for yr in range(2018, 2025):
        info(f"  python scripts/air/fetch_airnow_history.py --start-date {yr}-01-01 --end-date {yr}-12-31 --append")
    print()
    info("  Then run: python src/data/merge_datasets.py")
    info("  Then run: python src/features/build_features.py")
    info("  Then run: python src/models/train_models.py")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*62}")
    print("  SJV AQI PIPELINE DIAGNOSTICS")
    print(f"{'='*62}")

    check_queue_log()
    check_raw_air()
    check_processed_air()
    check_other_datasets()
    check_features()

    api_key = os.getenv("AIRNOW_API_KEY", "").strip()
    bbox_results = None
    if api_key:
        if CENTROIDS_PATH.exists():
            centroids = pd.read_csv(CENTROIDS_PATH)
            bbox_results = test_airnow_bbox(api_key, centroids)
        else:
            warn("Centroids file missing — skipping live API test.")
    else:
        hdr("6. Live AirNow API — Bbox Coverage Test")
        warn("AIRNOW_API_KEY not found in .env — skipping live API test.")
        info("Add your key to .env as AIRNOW_API_KEY=yourkey to run this check.")

    print_fix_prompts(bbox_results)

    print(f"\n{SEP}")
    print("  Diagnostics complete.")
    print(SEP)


if __name__ == "__main__":
    main()
