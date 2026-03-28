"""Residual plot — model error over time (test set).

Residual = predicted − actual.
Positive residuals = model over-predicted (said it'd be worse than it was).
Negative residuals = under-predicted (missed a bad air day).

Systematic clustering of residuals on certain dates reveals when the
model consistently fails — likely wildfire smoke events.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDS_PATH = PROJECT_ROOT / "results" / "models" / "regression_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "residuals.png"


def main() -> None:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDS_PATH}")

    df = pd.read_csv(PREDS_PATH)
    required = {"date", "target_next_day_aqi", "prediction_next_day_aqi"}
    if not required.issubset(df.columns):
        print(f"Predictions file missing columns: {required - set(df.columns)}")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["residual"] = df["prediction_next_day_aqi"] - df["target_next_day_aqi"]
    df = df.dropna(subset=["date", "residual"])

    if df.empty:
        print("No prediction rows available.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: residuals over time
    ax = axes[0]
    ax.scatter(df["date"], df["residual"], s=12, alpha=0.5, color="#1565C0", edgecolors="none")
    ax.axhline(0, color="black", linewidth=1)

    # Highlight large under-predictions (missed bad air days)
    bad_miss = df[df["residual"] < -30]
    ax.scatter(bad_miss["date"], bad_miss["residual"], s=30, color="#EF5350",
               edgecolors="none", label=f"Under-predicted >30 AQI units (n={len(bad_miss)})")

    # 30-day rolling mean residual
    daily_mean = df.groupby("date")["residual"].mean().sort_index()
    rolling = daily_mean.rolling("30D").mean()
    ax.plot(rolling.index, rolling.values, color="#FFA726", linewidth=1.8, label="30-day rolling mean residual")

    ax.set_ylabel("Residual (predicted − actual)")
    ax.set_title("Regression Model Residuals Over Time (Test Set)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom: residual distribution
    ax2 = axes[1]
    ax2.hist(df["residual"], bins=40, color="#42A5F5", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.axvline(df["residual"].mean(), color="#EF5350", linewidth=1.5, linestyle="--",
                label=f"Mean residual: {df['residual'].mean():.1f}")
    ax2.set_xlabel("Residual (predicted − actual AQI)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
