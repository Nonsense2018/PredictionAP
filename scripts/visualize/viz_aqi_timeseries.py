"""AQI time series for each SJV county (2018-2024).

Shows daily AQI mean with a 30-day rolling average overlay so seasonal
patterns and wildfire smoke spikes are easy to see.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "aqi_timeseries.png"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing air data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["aqi_mean"] = pd.to_numeric(df["aqi_mean"], errors="coerce")
    df = df.dropna(subset=["date", "aqi_mean"])

    if df.empty:
        print("No air quality data available yet. Run fetch_airnow_history.py first.")
        return

    counties = sorted(df["county"].unique())
    ncols = 2
    nrows = -(-len(counties) // ncols)  # ceiling division

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5), sharex=False)
    axes = axes.flatten()

    for i, county in enumerate(counties):
        ax = axes[i]
        county_df = df[df["county"] == county].sort_values("date")

        ax.plot(county_df["date"], county_df["aqi_mean"], color="#90CAF9", linewidth=0.6, alpha=0.7, label="Daily AQI")

        rolling = county_df.set_index("date")["aqi_mean"].rolling("30D").mean()
        ax.plot(rolling.index, rolling.values, color="#1565C0", linewidth=1.8, label="30-day avg")

        ax.axhline(100, color="#FFA726", linewidth=1, linestyle="--", alpha=0.8, label="Unhealthy for sensitive (100)")
        ax.axhline(150, color="#EF5350", linewidth=1, linestyle="--", alpha=0.8, label="Unhealthy (150)")

        ax.set_title(county, fontsize=11, fontweight="bold")
        ax.set_ylabel("AQI")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for j in range(len(counties), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9)
    fig.suptitle("Daily AQI — San Joaquin Valley Counties", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
