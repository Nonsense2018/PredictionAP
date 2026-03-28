"""Calendar heatmap of model prediction error (test set).

Each cell is the mean absolute error across all counties for that day.
Clusters of high error on specific dates reveal when the model consistently
fails — typically sudden wildfire smoke events or unseasonable pollution.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDS_PATH = PROJECT_ROOT / "results" / "models" / "regression_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "error_calendar.png"


def _calendar_array(series: pd.Series) -> tuple[np.ndarray, int, int]:
    """Convert a date-indexed series into a (week, weekday) array for one year."""
    year = series.index.year[0]
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    all_dates = pd.date_range(start, end)

    values = series.reindex(all_dates)

    # Week number within year (0-indexed), weekday (0=Mon, 6=Sun)
    week_nums = (all_dates - start).days // 7
    weekdays = all_dates.weekday

    n_weeks = week_nums.max() + 1
    grid = np.full((7, n_weeks), np.nan)
    for d, w, wd in zip(values, week_nums, weekdays):
        grid[wd, w] = d

    return grid, year, n_weeks


def main() -> None:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDS_PATH}")

    df = pd.read_csv(PREDS_PATH)
    required = {"date", "target_next_day_aqi", "prediction_next_day_aqi"}
    if not required.issubset(df.columns):
        print(f"Predictions file missing columns: {required - set(df.columns)}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["abs_error"] = (df["prediction_next_day_aqi"] - df["target_next_day_aqi"]).abs()
    df = df.dropna(subset=["date", "abs_error"])

    if df.empty:
        print("No prediction rows available.")
        return

    daily_mae = df.groupby("date")["abs_error"].mean()
    years = sorted(daily_mae.index.year.unique())

    fig, axes = plt.subplots(len(years), 1, figsize=(18, len(years) * 2.5))
    if len(years) == 1:
        axes = [axes]

    cmap = plt.cm.YlOrRd
    vmax = daily_mae.quantile(0.95)  # cap colorscale at 95th percentile for readability

    for ax, year in zip(axes, years):
        year_data = daily_mae[daily_mae.index.year == year]
        grid, _, n_weeks = _calendar_array(year_data)

        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest")

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_starts = []
        start = pd.Timestamp(f"{year}-01-01")
        for m in range(1, 13):
            day = pd.Timestamp(f"{year}-{m:02d}-01")
            month_starts.append((day - start).days // 7)

        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
        ax.set_title(f"{year} — Daily Mean Absolute Error (MAE) across counties", fontsize=10, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.8)
        cbar.set_label("MAE (AQI units)", fontsize=8)

    fig.suptitle("Calendar Heatmap: Regression Model Error (Test Set)\n"
                 "Bright cells = days the model was most wrong", fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
