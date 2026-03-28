"""Merge county-day air, meteorology, and wildfire datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIR_PATH = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
MET_PATH = PROJECT_ROOT / "data" / "processed" / "met" / "county_met_daily.csv"
FIRE_PATH = PROJECT_ROOT / "data" / "processed" / "fire" / "county_fire_daily.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "merged_daily_county.csv"


def _deduplicate_county_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate county/date rows by averaging numeric fields."""
    if frame.empty:
        return frame

    numeric_cols = [c for c in frame.columns if c not in {"county", "date"} and pd.api.types.is_numeric_dtype(frame[c])]
    non_numeric_cols = [c for c in frame.columns if c not in {"county", "date"} and c not in numeric_cols]

    aggregations: dict[str, str] = {col: "mean" for col in numeric_cols}
    aggregations.update({col: "first" for col in non_numeric_cols})

    return frame.groupby(["county", "date"], as_index=False).agg(aggregations)


def read_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """Load one processed dataset and validate merge keys."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {dataset_name} dataset: {path}")

    frame = pd.read_csv(path)
    required = {"county", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset_name} dataset missing columns: {sorted(missing)}")

    frame["county"] = frame["county"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["county", "date"]).copy()
    frame = _deduplicate_county_date(frame)
    return frame


def main() -> None:
    """Run dataset merge and save merged county-day table."""
    air = read_dataset(AIR_PATH, "air")
    met = read_dataset(MET_PATH, "meteorology")
    fire = read_dataset(FIRE_PATH, "wildfire")

    merged = air.merge(met, how="left", on=["county", "date"], suffixes=("", "_met"))
    merged = merged.merge(fire, how="left", on=["county", "date"], suffixes=("", "_fire"))

    if "fire_event_count_radius" in merged.columns:
        merged["fire_event_count_radius"] = merged["fire_event_count_radius"].fillna(0)
    if "smoke_present" in merged.columns:
        merged["smoke_present"] = merged["smoke_present"].fillna(0)
    if "min_fire_distance_km" in merged.columns:
        # NaN means no wildfire events were detected; fill with a large sentinel distance
        if "fire_radius_km" in merged.columns:
            sentinel = merged["fire_radius_km"].max() * 10
        else:
            sentinel = 9999.0
        merged["min_fire_distance_km"] = merged["min_fire_distance_km"].fillna(sentinel)

    merged = _deduplicate_county_date(merged)
    if merged.duplicated(subset=["county", "date"]).any():
        raise ValueError("Duplicate county/date rows detected after merge")

    merged = merged.sort_values(["county", "date"]).reset_index(drop=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote merged dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
