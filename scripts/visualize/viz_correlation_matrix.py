"""Feature correlation matrix.

Shows pairwise Pearson correlations between all numeric features and the
regression target. Surfaces which variables are predictive and which are
redundant with each other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "features_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "correlation_matrix.png"

# Friendlier display names
RENAME = {
    "aqi_mean": "AQI (today)",
    "aqi_lag_1": "AQI lag 1d",
    "aqi_lag_2": "AQI lag 2d",
    "aqi_lag_3": "AQI lag 3d",
    "aqi_roll3_mean": "AQI roll 3d",
    "aqi_roll7_mean": "AQI roll 7d",
    "temperature_2m_mean": "Temp mean",
    "temperature_2m_max": "Temp max",
    "temperature_2m_min": "Temp min",
    "precipitation_sum": "Precip",
    "wind_speed_10m_max": "Wind max",
    "fire_event_count_radius": "Fire count",
    "smoke_present": "Smoke",
    "min_fire_distance_km": "Fire dist (km)",
    "target_next_day_aqi": "TARGET: next AQI",
}


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing feature dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    keep = [c for c in RENAME if c in df.columns]
    if not keep:
        print("No expected feature columns found.")
        return

    sub = df[keep].rename(columns=RENAME)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # Only mask strict upper triangle (keep diagonal)
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            mask.iloc[i, j] = True

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
