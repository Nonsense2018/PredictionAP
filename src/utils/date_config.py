"""Shared date configuration loading for data collection and model splits."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATES_CONFIG_PATH = PROJECT_ROOT / "configs" / "dates_config.json"


def parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a date."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_dates_config(config_path: Optional[Path] = None) -> dict[str, str]:
    """Load raw date config JSON as a dictionary."""
    path = config_path or DEFAULT_DATES_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing dates config file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("dates_config.json must be a JSON object")
    return payload


def get_data_collection_range(config_path: Optional[Path] = None) -> tuple[date, date]:
    """Get global start/end range for data collection."""
    payload = load_dates_config(config_path)
    start_date = parse_iso_date(payload["start_date"])
    end_date = parse_iso_date(payload["end_date"])

    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    return start_date, end_date


def get_train_val_test_ranges(config_path: Optional[Path] = None) -> dict[str, tuple[date, date]]:
    """Get strict train/val/test date windows."""
    payload = load_dates_config(config_path)

    train = (parse_iso_date(payload["train_start"]), parse_iso_date(payload["train_end"]))
    val = (parse_iso_date(payload["val_start"]), parse_iso_date(payload["val_end"]))
    test = (parse_iso_date(payload["test_start"]), parse_iso_date(payload["test_end"]))

    if not (train[0] <= train[1] < val[0] <= val[1] < test[0] <= test[1]):
        raise ValueError("Train/val/test ranges must be ordered and non-overlapping")

    return {"train": train, "val": val, "test": test}
