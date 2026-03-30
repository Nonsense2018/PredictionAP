"""Create modeling features and next-day targets from merged county-day data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "merged_daily_county.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "features_dataset.csv"
EXCEEDANCE_THRESHOLD = 100  # AQI >= 100 (EPA "Unhealthy for Sensitive Groups" — school outdoor activity restriction proxy)


def main() -> None:
    """Build lag/rolling features and next-day targets."""
    if not MERGED_PATH.exists():
        raise FileNotFoundError(f"Missing merged dataset: {MERGED_PATH}")

    df = pd.read_csv(MERGED_PATH)
    required_columns = {"county", "date", "aqi_mean"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Merged dataset missing columns: {sorted(missing)}")

    df["county"] = df["county"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["aqi_mean"] = pd.to_numeric(df["aqi_mean"], errors="coerce")
    df = df.dropna(subset=["county", "date", "aqi_mean"]).copy()
    df = df.sort_values(["county", "date"]).reset_index(drop=True)

    grouped = df.groupby("county", group_keys=False)

    df["aqi_lag_1"] = grouped["aqi_mean"].shift(1)
    df["aqi_lag_2"] = grouped["aqi_mean"].shift(2)
    df["aqi_lag_3"] = grouped["aqi_mean"].shift(3)

    df["aqi_roll3_mean"] = grouped["aqi_mean"].shift(1).rolling(window=3).mean().reset_index(level=0, drop=True)
    df["aqi_roll7_mean"] = grouped["aqi_mean"].shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)

    df["target_next_day_aqi"] = grouped["aqi_mean"].shift(-1)
    df["target_next_day_exceedance"] = (df["target_next_day_aqi"] >= EXCEEDANCE_THRESHOLD).astype("float")

    # Keep rows where targets and core lag features are available.
    feature_columns = ["aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_roll3_mean", "aqi_roll7_mean"]
    keep_mask = df["target_next_day_aqi"].notna()
    keep_mask &= df[feature_columns].notna().all(axis=1)
    df = df.loc[keep_mask].copy()

    df["date"] = df["date"].dt.date
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote feature dataset: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
