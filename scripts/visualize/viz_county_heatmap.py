"""County × month AQI heatmap.

Each cell is the mean AQI for that county in that calendar month, averaged
across all years. Reveals which counties are worst and which months are worst.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "air" / "county_air_daily.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "county_month_heatmap.png"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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

    df["month"] = df["date"].dt.month

    pivot = (
        df.groupby(["county", "month"])["aqi_mean"]
        .mean()
        .unstack(level="month")
        .reindex(columns=range(1, 13))
    )
    pivot.columns = MONTH_LABELS

    # Sort counties by annual mean AQI descending (worst at top)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        cbar_kws={"label": "Mean AQI"},
        vmin=0,
        vmax=150,
    )
    ax.set_title("Mean AQI by County and Month (all years combined)", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
