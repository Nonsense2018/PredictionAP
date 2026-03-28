"""Basic setup checks before running the full pipeline."""

from __future__ import annotations

import importlib
from pathlib import Path

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODULES = [
    "pandas",
    "requests",
    "dotenv",
    "geopandas",
    "sklearn",
    "joblib",
]
REQUIRED_PATHS = [
    PROJECT_ROOT / "configs" / "counties_sjv.json",
    PROJECT_ROOT / "configs" / "dates_config.json",
    PROJECT_ROOT / "data" / "raw" / "geo",
    PROJECT_ROOT / "scripts" / "geo" / "get_county_centroids.py",
    PROJECT_ROOT / "scripts" / "air" / "fetch_airnow_history.py",
    PROJECT_ROOT / "scripts" / "met" / "fetch_openmeteo_history.py",
    PROJECT_ROOT / "scripts" / "fire" / "fetch_eonet_wildfire_data.py",
]


def check_python_modules() -> list[str]:
    """Return a list of missing Python modules."""
    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def check_paths() -> list[str]:
    """Return a list of required paths that do not exist."""
    missing = []
    for path in REQUIRED_PATHS:
        if not path.exists():
            missing.append(str(path))
    return missing


def check_env() -> list[str]:
    """Validate required env vars."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    missing = []
    if not os.getenv("AIRNOW_API_KEY", "").strip():
        missing.append("AIRNOW_API_KEY")
    return missing


def main() -> None:
    """Run all setup checks and print a compact report."""
    module_missing = check_python_modules()
    path_missing = check_paths()
    env_missing = check_env()

    if module_missing:
        print("Missing Python modules:")
        for item in module_missing:
            print(f"- {item}")

    if path_missing:
        print("Missing required paths:")
        for item in path_missing:
            print(f"- {item}")

    if env_missing:
        print("Missing required env vars:")
        for item in env_missing:
            print(f"- {item}")

    if not module_missing and not path_missing and not env_missing:
        print("Setup check passed.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
