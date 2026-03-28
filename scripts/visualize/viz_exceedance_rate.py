"""Exceedance rate by county — percentage of days AQI >= 120.

Shows which communities face the most health-risk days per year.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "exceedance_rate.png"

EXCEEDANCE_THRESHOLD = 120


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

    df["year"] = df["date"].dt.year
    df["exceeded"] = (df["aqi_mean"] >= EXCEEDANCE_THRESHOLD).astype(int)

    # Per-county per-year exceedance rate
    rates = (
        df.groupby(["county", "year"])["exceeded"]
        .mean()
        .mul(100)
        .reset_index(name="exceedance_pct")
    )

    counties = sorted(rates["county"].unique())
    years = sorted(rates["year"].unique())
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: grouped bar chart by year
    ax = axes[0]
    width = 0.8 / len(years)
    for j, year in enumerate(years):
        year_data = rates[rates["year"] == year].set_index("county").reindex(counties)
        x = [i + j * width for i in range(len(counties))]
        ax.bar(x, year_data["exceedance_pct"].fillna(0), width=width,
               label=str(year), color=colors[j % len(colors)], alpha=0.85)

    ax.set_xticks([i + width * (len(years) - 1) / 2 for i in range(len(counties))])
    ax.set_xticklabels(counties, rotation=30, ha="right")
    ax.set_ylabel("% Days with AQI ≥ 120")
    ax.set_title(f"Annual Exceedance Rate by County (AQI ≥ {EXCEEDANCE_THRESHOLD})", fontweight="bold")
    ax.legend(title="Year", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: overall mean exceedance rate per county (sorted)
    ax2 = axes[1]
    overall = rates.groupby("county")["exceedance_pct"].mean().sort_values(ascending=True)
    bars = ax2.barh(overall.index, overall.values, color="#EF5350", alpha=0.85)
    ax2.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax2.set_xlabel("Mean % Days with AQI ≥ 120 (all years)")
    ax2.set_title("Overall Exceedance Rate by County", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
