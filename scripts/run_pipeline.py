"""Run the full sjv-aqi-risk pipeline in the required step order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Scripts that accept --start-date / --end-date overrides
DATE_AWARE_STEPS = {
    "scripts/air/fetch_airnow_history.py",
    "scripts/met/fetch_openmeteo_history.py",
    "scripts/fire/fetch_eonet_wildfire_data.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full SJV AQI risk pipeline")
    parser.add_argument("--start-date", type=str, default=None, help="Override collection start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Override collection end date (YYYY-MM-DD)")
    parser.add_argument("--append-air", action="store_true", help="Append to existing AirNow raw records instead of overwriting")
    return parser.parse_args()


def run_step(relative_script: str, extra_args: list[str] | None = None) -> None:
    """Execute one Python script and stop on failure."""
    script_path = PROJECT_ROOT / relative_script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing step script: {script_path}")

    cmd = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"\n=== Running: {relative_script} ===")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Run pipeline from centroid generation through model training."""
    args = parse_args()

    date_args: list[str] = []
    if args.start_date:
        date_args += ["--start-date", args.start_date]
    if args.end_date:
        date_args += ["--end-date", args.end_date]

    steps = [
        "scripts/geo/get_county_centroids.py",
        "scripts/air/fetch_airnow_history.py",
        "scripts/met/fetch_openmeteo_history.py",
        "scripts/fire/fetch_eonet_wildfire_data.py",
        "src/data/merge_datasets.py",
        "src/features/build_features.py",
        "src/models/train_models.py",
    ]

    for step in steps:
        extra: list[str] = []
        if step in DATE_AWARE_STEPS:
            extra += date_args
        if step == "scripts/air/fetch_airnow_history.py" and args.append_air:
            extra += ["--append"]
        run_step(step, extra or None)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
