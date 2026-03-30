"""
Queue runner: waits for 2018 AirNow data to finish, then runs 2019-2024 sequentially.
Run this while the 2018 fetch is already in progress.
"""
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIR_PROCESSED = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
AIR_SCRIPT = PROJECT_ROOT / "scripts" / "air" / "fetch_airnow_history.py"

REMAINING_YEARS = [
    ("2019-01-01", "2019-12-31"),
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
]

def year_is_done(year: int) -> bool:
    """Check if the processed file contains rows from the given year."""
    if not AIR_PROCESSED.exists():
        return False
    import pandas as pd
    try:
        df = pd.read_csv(AIR_PROCESSED, usecols=["date"])
        return any(str(year) in str(d) for d in df["date"].dropna())
    except Exception:
        return False

def wait_for_2018(poll_interval: int = 120) -> None:
    print("Waiting for 2018 AirNow fetch to complete...", flush=True)
    while not year_is_done(2018):
        print(f"  2018 not yet complete. Checking again in {poll_interval}s...", flush=True)
        time.sleep(poll_interval)
    print("2018 data confirmed present. Starting 2019-2024 queue.", flush=True)

def run_year(start_date: str, end_date: str) -> int:
    year = start_date[:4]
    print(f"\n{'='*60}", flush=True)
    print(f"Starting AirNow fetch: {start_date} to {end_date}", flush=True)
    print(f"{'='*60}", flush=True)
    cmd = [sys.executable, str(AIR_SCRIPT),
           "--start-date", start_date,
           "--end-date", end_date,
           "--append"]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: Year {year} exited with code {result.returncode}. Continuing anyway.", flush=True)
    return result.returncode

def main() -> None:
    wait_for_2018()
    for start_date, end_date in REMAINING_YEARS:
        year = start_date[:4]
        if year_is_done(int(year)):
            print(f"Year {year} already present in data — skipping.", flush=True)
            continue
        run_year(start_date, end_date)
    print("\nAll AirNow years complete!", flush=True)

if __name__ == "__main__":
    main()
