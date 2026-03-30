"""
Re-fetch AirNow data for Kern, Kings, and Madera using wider bounding boxes.
Temporarily swaps the centroids file to only include the target county,
then restores it after each run to avoid duplicating existing county data.
Runs sequentially: Kern -> Madera -> Kings.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CENTROIDS_PATH = PROJECT_ROOT / "data" / "processed" / "geo" / "counties_centroids.csv"
FETCH_SCRIPT = PROJECT_ROOT / "scripts" / "air" / "fetch_airnow_history.py"
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "missing_counties.log"

COUNTIES_TO_FETCH = [
    ("Kern",   0.5),
    ("Madera", 0.5),
    ("Kings",  1.0),
]


def run_county(county: str, bbox: float, centroids_backup: pd.DataFrame) -> int:
    # Write single-county centroids file
    single = centroids_backup[centroids_backup["county"] == county]
    single.to_csv(CENTROIDS_PATH, index=False)
    print(f"\n{'='*60}", flush=True)
    print(f"Fetching {county} (bbox +/-{bbox} degrees, 2018-2024)", flush=True)
    print(f"{'='*60}", flush=True)

    bbox_arg = json.dumps({county: bbox})
    cmd = [
        sys.executable, str(FETCH_SCRIPT),
        "--start-date", "2018-01-01",
        "--end-date",   "2024-12-31",
        "--append",
        "--county-bbox", bbox_arg,
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: {county} exited with code {result.returncode}. Continuing.", flush=True)
    return result.returncode


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    centroids_backup = pd.read_csv(CENTROIDS_PATH)

    try:
        for county, bbox in COUNTIES_TO_FETCH:
            run_county(county, bbox, centroids_backup)
    finally:
        # Always restore full centroids file, even if interrupted
        centroids_backup.to_csv(CENTROIDS_PATH, index=False)
        print("\nCentroids file restored.", flush=True)

    print("\nAll three counties fetched. Run the pipeline next:", flush=True)
    print("  python src/data/merge_datasets.py", flush=True)
    print("  python src/features/build_features.py", flush=True)
    print("  python src/models/train_models.py", flush=True)
    print("  python scripts/visualize/run_all_visualizations.py", flush=True)
    print("  python scripts/diagnostics.py", flush=True)


if __name__ == "__main__":
    main()
