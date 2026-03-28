"""Wildfire proximity vs AQI scatter plot.

Plots min_fire_distance_km against aqi_mean for all county-days that had
at least one wildfire event nearby. Reveals how strongly fire proximity
drives AQI spikes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "merged_daily_county.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "wildfire_aqi_scatter.png"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing merged data: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["aqi_mean"] = pd.to_numeric(df["aqi_mean"], errors="coerce")
    df["min_fire_distance_km"] = pd.to_numeric(df["min_fire_distance_km"], errors="coerce")
    df["fire_event_count_radius"] = pd.to_numeric(df["fire_event_count_radius"], errors="coerce")

    if df.empty or df["aqi_mean"].dropna().empty:
        print("No merged data available yet. Run the pipeline first.")
        return

    # Only days where at least one fire was detected (real distance, not sentinel)
    fire_days = df[(df["fire_event_count_radius"] > 0) & df["min_fire_distance_km"].notna()].copy()

    if fire_days.empty:
        print("No wildfire event days found in the data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: scatter, one point per county-day with fire activity
    ax = axes[0]
    scatter = ax.scatter(
        fire_days["min_fire_distance_km"],
        fire_days["aqi_mean"],
        c=fire_days["fire_event_count_radius"],
        cmap="YlOrRd",
        alpha=0.5,
        s=15,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("# fires within radius")
    ax.set_xlabel("Distance to nearest fire (km)")
    ax.set_ylabel("AQI mean")
    ax.set_title("Wildfire Proximity vs AQI", fontweight="bold")
    ax.axhline(120, color="#EF5350", linewidth=1, linestyle="--", alpha=0.7, label="Exceedance threshold (120)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: AQI distribution on fire days vs non-fire days
    ax2 = axes[1]
    no_fire = df[df["fire_event_count_radius"] == 0]["aqi_mean"].dropna()
    has_fire = fire_days["aqi_mean"].dropna()

    bins = np.linspace(0, df["aqi_mean"].quantile(0.99), 40)
    ax2.hist(no_fire, bins=bins, alpha=0.6, color="#42A5F5", label=f"No nearby fire (n={len(no_fire):,})", density=True)
    ax2.hist(has_fire, bins=bins, alpha=0.6, color="#EF5350", label=f"Fire within radius (n={len(has_fire):,})", density=True)
    ax2.axvline(120, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax2.set_xlabel("AQI mean")
    ax2.set_ylabel("Density")
    ax2.set_title("AQI Distribution: Fire Days vs Non-Fire Days", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
