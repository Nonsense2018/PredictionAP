"""
Fit a Prophet model to Fresno AQI and decompose into trend + yearly seasonality.
Produces two plots:
  1. results/plots/prophet_seasonality.png  — trend + yearly seasonal component
  2. results/plots/prophet_forecast.png     — actual vs Prophet forecast with uncertainty
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from prophet import Prophet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "features_dataset.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
COUNTY = "Fresno"


def load_fresno() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df[df["county"] == COUNTY][["date", "aqi_mean"]].dropna()
    df = df.rename(columns={"date": "ds", "aqi_mean": "y"})
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_fresno()

    # Train on 2018-2023, leave 2024 for visual comparison
    train = df[df["ds"] < "2024-01-01"]
    actual_2024 = df[df["ds"] >= "2024-01-01"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    model.fit(train)

    # Forecast through end of 2024
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # --- Plot 1: Seasonality decomposition ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # Trend
    axes[0].plot(forecast["ds"], forecast["trend"], color="#2c7bb6", linewidth=1.5)
    axes[0].set_title(f"{COUNTY} — Long-Term AQI Trend (Prophet)", fontsize=13)
    axes[0].set_ylabel("AQI (trend component)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].grid(axis="y", alpha=0.3)

    # Yearly seasonality — extract one year's worth of points
    seasonality = forecast[["ds", "yearly"]].copy()
    seasonality["month_day"] = seasonality["ds"].dt.strftime("%m-%d")
    seasonal_mean = (
        seasonality.groupby("month_day")["yearly"]
        .mean()
        .reset_index()
        .sort_values("month_day")
    )
    # Convert month_day to a date for x-axis labelling
    seasonal_mean["date"] = pd.to_datetime("2023-" + seasonal_mean["month_day"], errors="coerce")
    seasonal_mean = seasonal_mean.dropna(subset=["date"])

    axes[1].fill_between(
        seasonal_mean["date"], seasonal_mean["yearly"],
        alpha=0.35, color="#d7191c"
    )
    axes[1].plot(seasonal_mean["date"], seasonal_mean["yearly"], color="#d7191c", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title(f"{COUNTY} — Yearly AQI Seasonality (wildfire season visible Aug–Oct)", fontsize=13)
    axes[1].set_ylabel("AQI (seasonal component)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out1 = PLOTS_DIR / "prophet_seasonality.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # --- Plot 2: Actual vs forecast with uncertainty interval ---
    fig, ax = plt.subplots(figsize=(14, 5))

    # Uncertainty band
    ax.fill_between(
        forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
        alpha=0.2, color="#2c7bb6", label="95% uncertainty interval"
    )
    # Prophet forecast line
    ax.plot(forecast["ds"], forecast["yhat"], color="#2c7bb6", linewidth=1.2, label="Prophet forecast")
    # Training actuals
    ax.scatter(train["ds"], train["y"], s=2, color="black", alpha=0.4, label="Actual AQI (train)")
    # 2024 actuals
    ax.scatter(actual_2024["ds"], actual_2024["y"], s=6, color="#d7191c", alpha=0.7, label="Actual AQI (2024 holdout)")
    # Threshold line
    ax.axhline(100, color="#f4a11d", linewidth=1.2, linestyle="--", label="Exceedance threshold (AQI 100)")

    ax.set_title(f"{COUNTY} — Prophet AQI Forecast vs Actual (2018–2024)", fontsize=13)
    ax.set_ylabel("AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out2 = PLOTS_DIR / "prophet_forecast.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
