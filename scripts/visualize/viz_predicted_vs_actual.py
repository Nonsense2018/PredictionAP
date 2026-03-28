"""Predicted vs actual AQI scatter (regression model, test set).

Points above the diagonal = model over-predicted.
Points below = under-predicted.
Color by county to spot county-specific biases.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDS_PATH = PROJECT_ROOT / "results" / "models" / "regression_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "predicted_vs_actual.png"


def main() -> None:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDS_PATH}")

    df = pd.read_csv(PREDS_PATH)
    required = {"target_next_day_aqi", "prediction_next_day_aqi"}
    if not required.issubset(df.columns):
        print(f"Predictions file missing columns: {required - set(df.columns)}")
        return

    df = df.dropna(subset=list(required))
    if df.empty:
        print("No prediction rows available.")
        return

    counties = sorted(df["county"].unique()) if "county" in df.columns else []
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 7))

    if counties:
        for i, county in enumerate(counties):
            sub = df[df["county"] == county]
            ax.scatter(sub["target_next_day_aqi"], sub["prediction_next_day_aqi"],
                       label=county, color=colors[i % len(colors)], s=20, alpha=0.7, edgecolors="none")
    else:
        ax.scatter(df["target_next_day_aqi"], df["prediction_next_day_aqi"],
                   s=20, alpha=0.7, edgecolors="none")

    # Perfect prediction line
    lo = min(df["target_next_day_aqi"].min(), df["prediction_next_day_aqi"].min()) * 0.95
    hi = max(df["target_next_day_aqi"].max(), df["prediction_next_day_aqi"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect prediction")

    # Compute metrics inline for annotation
    mae = float(np.abs(df["target_next_day_aqi"] - df["prediction_next_day_aqi"]).mean())
    rmse = float(np.sqrt(((df["target_next_day_aqi"] - df["prediction_next_day_aqi"]) ** 2).mean()))
    ss_res = ((df["target_next_day_aqi"] - df["prediction_next_day_aqi"]) ** 2).sum()
    ss_tot = ((df["target_next_day_aqi"] - df["target_next_day_aqi"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    ax.text(0.04, 0.96, f"MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nR²: {r2:.3f}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10)

    ax.set_xlabel("Actual next-day AQI")
    ax.set_ylabel("Predicted next-day AQI")
    ax.set_title("Predicted vs Actual AQI (Test Set — Regression Model)", fontweight="bold")
    ax.legend(fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
