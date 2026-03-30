"""
Master visualization script for SJV AQI Risk Prediction.
Generates all plots and organizes them into per-model and comparison folders.

Usage:
    python scripts/visualize/run_all_visualizations.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH  = PROJECT_ROOT / "data" / "processed" / "modeling" / "features_dataset.csv"
MERGED_PATH    = PROJECT_ROOT / "data" / "processed" / "modeling" / "merged_daily_county.csv"
METRICS_PATH   = PROJECT_ROOT / "results" / "models" / "metrics.csv"
CLS_PREDS_PATH = PROJECT_ROOT / "results" / "models" / "classification_predictions.csv"
REG_PREDS_PATH = PROJECT_ROOT / "results" / "models" / "regression_predictions.csv"
MODELS_DIR     = PROJECT_ROOT / "results" / "models"
PLOTS_ROOT     = PROJECT_ROOT / "results" / "plots"

EXCEEDANCE_THRESHOLD = 100

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = {
    "ridge":         "#4C72B0",
    "random_forest": "#55A868",
    "xgboost":       "#C44E52",
    "lightgbm":      "#8172B2",
    "logistic":      "#CCB974",
    "persistence":   "#888888",
    "prophet":       "#64B5CD",
}
MODEL_LABELS = {
    "ridge":         "Ridge",
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
    "logistic":      "Logistic",
    "persistence":   "Persistence",
    "prophet":       "Prophet",
}
COUNTY_COLORS = {
    "Fresno":      "#E63946",
    "Kern":        "#F4A261",
    "Kings":       "#2A9D8F",
    "Madera":      "#457B9D",
    "Merced":      "#A8DADC",
    "San Joaquin": "#6A0572",
    "Stanislaus":  "#E9C46A",
    "Tulare":      "#264653",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#CCCCCC",
        "axes.linewidth":   0.8,
        "axes.grid":        True,
        "grid.color":       "#EEEEEE",
        "grid.linewidth":   0.7,
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "legend.fontsize":  9,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#CCCCCC",
    })

def _save(fig, path: Path, tight: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA OVERVIEW PLOTS  →  all_models/
# ══════════════════════════════════════════════════════════════════════════════

def plot_aqi_timeseries(df: pd.DataFrame):
    """County AQI time series with 30-day rolling mean and exceedance line."""
    counties = sorted(df["county"].unique())
    ncols = 2
    nrows = (len(counties) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2), sharex=True)
    axes = axes.flatten()

    for i, county in enumerate(counties):
        ax = axes[i]
        cdf = df[df["county"] == county].copy()
        cdf["date"] = pd.to_datetime(cdf["date"])
        cdf = cdf.sort_values("date")

        color = COUNTY_COLORS.get(county, "#555555")
        ax.fill_between(cdf["date"], cdf["aqi_mean"], alpha=0.15, color=color)
        ax.plot(cdf["date"], cdf["aqi_mean"], linewidth=0.6, color=color, alpha=0.5, label="Daily AQI")
        roll = cdf["aqi_mean"].rolling(30, center=True).mean()
        ax.plot(cdf["date"], roll, linewidth=1.8, color=color, label="30-day avg")
        ax.axhline(EXCEEDANCE_THRESHOLD, color="#E63946", linewidth=1.1,
                   linestyle="--", alpha=0.7, label=f"AQI {EXCEEDANCE_THRESHOLD}")
        ax.set_title(county)
        ax.set_ylabel("AQI")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        if i == 0:
            ax.legend(loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Daily AQI by County — San Joaquin Valley (2018–2024)",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "aqi_timeseries.png")


def plot_county_heatmap(df: pd.DataFrame):
    """County × month mean AQI heatmap averaged across all years."""
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2["month"] = df2["date"].dt.month
    pivot = df2.pivot_table(index="county", columns="month", values="aqi_mean", aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
        linewidths=0.4, linecolor="#DDDDDD",
        cbar_kws={"label": "Mean AQI", "shrink": 0.8},
    )
    ax.set_title("Mean AQI by County and Month (2018–2024)", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "county_month_heatmap.png")


def plot_exceedance_by_year(df: pd.DataFrame):
    """Stacked bar: exceedance days per year per county."""
    df2 = df.copy()
    df2["year"] = pd.to_datetime(df2["date"]).dt.year
    exc = df2[df2["target_next_day_exceedance"] == 1]
    pivot = exc.groupby(["year", "county"]).size().unstack(fill_value=0)

    counties = pivot.columns.tolist()
    colors = [COUNTY_COLORS.get(c, "#999") for c in counties]

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.65, edgecolor="white")
    ax.set_title(f"Days with AQI ≥ {EXCEEDANCE_THRESHOLD} by Year and County", pad=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Exceedance Days")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="County", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "exceedance_by_year.png")


def plot_correlation_matrix(df: pd.DataFrame):
    """Feature correlation matrix with the regression target."""
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and c not in {"target_next_day_exceedance"}
                and df[c].notna().sum() > 100]
    corr = df[num_cols].corr()

    # Rename for readability
    rename = {
        "aqi_mean": "AQI (today)",
        "aqi_lag_1": "AQI lag 1",
        "aqi_lag_2": "AQI lag 2",
        "aqi_lag_3": "AQI lag 3",
        "aqi_roll3_mean": "AQI roll 3d",
        "aqi_roll7_mean": "AQI roll 7d",
        "target_next_day_aqi": "AQI (tomorrow)",
        "temperature_2m_mean": "Temp mean",
        "fire_event_count_radius": "Fire events",
    }
    corr = corr.rename(index=rename, columns=rename)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, ax=ax, mask=mask, cmap="coolwarm", center=0,
        annot=True, fmt=".2f", linewidths=0.4, linecolor="#EEEEEE",
        vmin=-1, vmax=1,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", pad=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "correlation_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON PLOTS  →  all_models/
# ══════════════════════════════════════════════════════════════════════════════

def plot_regression_comparison(metrics: pd.DataFrame):
    """Grouped bar chart comparing MAE and RMSE across regression models."""
    reg = metrics[(metrics["task"] == "regression") & (metrics["split"] == "test")].copy()
    reg = reg[reg["model"] != "prophet"]  # prophet is time-series only, apples/oranges
    reg["label"] = reg["model"].map(MODEL_LABELS)
    reg = reg.sort_values("mae")

    x = np.arange(len(reg))
    w = 0.35
    colors = [PALETTE.get(m, "#999") for m in reg["model"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, reg["mae"], w, label="MAE", color=colors, alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + w/2, reg["rmse"], w, label="RMSE", color=colors, alpha=0.5,
                   edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(reg["label"], rotation=0)
    ax.set_ylabel("Error (AQI units)")
    ax.set_title("Regression Model Comparison — Test Set (MAE & RMSE)", pad=10)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "model_comparison_regression.png")


def plot_classification_comparison(metrics: pd.DataFrame):
    """Grouped bar: Precision, Recall, F1, ROC-AUC for all classifiers (test set)."""
    cls = metrics[(metrics["task"] == "classification") & (metrics["split"] == "test")].copy()
    cls["label"] = cls["model"].map(MODEL_LABELS)
    cls = cls.sort_values("f1", ascending=False)

    metric_cols = ["precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(cls))
    w = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(13, 5))
    for offset, col, label, color in zip(offsets, metric_cols, metric_labels, bar_colors):
        vals = cls[col].fillna(0).values
        bars = ax.bar(x + offset * w, vals, w, label=label, color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cls["label"], rotation=0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Classification Model Comparison — Test Set", pad=10)
    ax.legend(loc="upper right")
    ax.axhline(1.0, color="#CCCCCC", linewidth=0.7, linestyle="--")
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "model_comparison_classification.png")


def plot_roc_all(cls_preds: pd.DataFrame):
    """All ROC curves on one plot."""
    actual = cls_preds["target_next_day_exceedance"].astype(int)
    if actual.nunique() < 2:
        print("  Skipping ROC: only one class in test set.")
        return

    prob_cols = {
        "Logistic":      "prob_exc_logistic",
        "Random Forest": "prediction_probability_exceedance",
        "XGBoost":       "prob_exc_xgboost",
        "LightGBM":      "prob_exc_lightgbm",
    }
    model_keys = ["logistic", "random_forest", "xgboost", "lightgbm"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, alpha=0.5, label="Random (AUC = 0.50)")

    for name, key, prob_col in zip(prob_cols.keys(), model_keys, prob_cols.values()):
        if prob_col not in cls_preds.columns:
            continue
        fpr, tpr, _ = roc_curve(actual, cls_preds[prob_col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=PALETTE[key],
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — All Classifiers (Test Set, AQI ≥ {EXCEEDANCE_THRESHOLD})", pad=10)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "roc_curves_all.png")


def plot_residuals_all(reg_preds: pd.DataFrame):
    """Residual (predicted − actual) over time for all regression models."""
    reg_preds = reg_preds.copy()
    reg_preds["date"] = pd.to_datetime(reg_preds["date"])

    model_cols = {
        "Ridge":         "pred_aqi_ridge",
        "Random Forest": "prediction_next_day_aqi",
        "XGBoost":       "pred_aqi_xgboost",
        "LightGBM":      "pred_aqi_lightgbm",
        "Persistence":   "pred_aqi_persistence",
    }
    model_keys = ["ridge", "random_forest", "xgboost", "lightgbm", "persistence"]

    fig, axes = plt.subplots(len(model_cols), 1, figsize=(14, 12), sharex=True)
    for ax, (label, col), key in zip(axes, model_cols.items(), model_keys):
        if col not in reg_preds.columns:
            ax.set_visible(False)
            continue
        residual = reg_preds[col] - reg_preds["target_next_day_aqi"]
        ax.axhline(0, color="#999999", linewidth=0.9, linestyle="--")
        ax.scatter(reg_preds["date"], residual, s=3, alpha=0.4, color=PALETTE[key])
        roll = residual.rolling(30).mean()
        ax.plot(reg_preds["date"].values, roll.values, color=PALETTE[key], linewidth=1.5)
        mae = residual.abs().mean()
        ax.set_ylabel("Residual\n(pred − actual)", fontsize=9)
        ax.set_title(f"{label}  (Test MAE = {mae:.1f})", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    axes[-1].tick_params(axis="x", rotation=30)
    fig.suptitle("Regression Residuals Over Time — Test Set (2024)", fontsize=14,
                 fontweight="bold", y=1.005)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "all_models" / "residuals_all_models.png")


# ══════════════════════════════════════════════════════════════════════════════
# PER-MODEL PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(reg_preds: pd.DataFrame, model_key: str, model_label: str, pred_col: str):
    """Predicted vs actual scatter for one regression model, colored by county."""
    sub = reg_preds[["county", "target_next_day_aqi", pred_col]].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for county, group in sub.groupby("county"):
        ax.scatter(group["target_next_day_aqi"], group[pred_col],
                   s=12, alpha=0.55, color=COUNTY_COLORS.get(county, "#999"),
                   label=county, edgecolors="none")

    lims = [min(sub["target_next_day_aqi"].min(), sub[pred_col].min()) - 5,
            max(sub["target_next_day_aqi"].max(), sub[pred_col].max()) + 5]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Perfect prediction")
    ax.axvline(EXCEEDANCE_THRESHOLD, color="#E63946", linewidth=0.9,
               linestyle=":", alpha=0.7, label=f"AQI {EXCEEDANCE_THRESHOLD}")
    ax.axhline(EXCEEDANCE_THRESHOLD, color="#E63946", linewidth=0.9, linestyle=":", alpha=0.7)

    mae = (sub[pred_col] - sub["target_next_day_aqi"]).abs().mean()
    ax.set_xlabel("Actual Next-Day AQI")
    ax.set_ylabel("Predicted Next-Day AQI")
    ax.set_title(f"{model_label} — Predicted vs Actual (Test Set)\nMAE = {mae:.1f}", pad=10)
    ax.legend(loc="upper left", markerscale=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / model_key / "predicted_vs_actual.png")


def plot_confusion_matrix_model(cls_preds: pd.DataFrame, model_key: str, model_label: str, pred_col: str):
    """Confusion matrix for one classifier."""
    sub = cls_preds[["target_next_day_exceedance", pred_col]].dropna()
    actual = sub["target_next_day_exceedance"].astype(int)
    pred   = sub[pred_col].astype(int)

    cm = confusion_matrix(actual, pred)
    labels = [f"AQI < {EXCEEDANCE_THRESHOLD}\n(No Risk)", f"AQI ≥ {EXCEEDANCE_THRESHOLD}\n(Exceedance)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, ax=ax, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#DDDDDD",
                cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", labelpad=8)
    ax.set_ylabel("Actual", labelpad=8)
    prec = precision_score(actual, pred, zero_division=0)
    rec  = recall_score(actual, pred, zero_division=0)
    f1   = f1_score(actual, pred, zero_division=0)
    ax.set_title(f"{model_label} — Confusion Matrix (Test Set)\n"
                 f"Precision={prec:.2f}  Recall={rec:.2f}  F1={f1:.2f}", pad=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / model_key / "confusion_matrix.png")


def plot_feature_importance_model(model_path: Path, feature_cols: list[str],
                                   model_key: str, model_label: str, task: str):
    """Feature importance bar chart for tree-based models."""
    if not model_path.exists():
        return
    model = joblib.load(model_path)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return

    rename = {
        "aqi_lag_1": "AQI lag 1d",
        "aqi_lag_2": "AQI lag 2d",
        "aqi_lag_3": "AQI lag 3d",
        "aqi_roll3_mean": "AQI roll 3d",
        "aqi_roll7_mean": "AQI roll 7d",
        "temperature_2m_mean": "Temp mean",
        "fire_event_count_radius": "Fire events",
    }
    labels = [rename.get(c, c) for c in feature_cols]
    idx = np.argsort(importances)[::-1]
    sorted_imp = importances[idx]
    sorted_labels = [labels[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(sorted_labels[::-1], sorted_imp[::-1],
                   color=PALETTE[model_key], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, sorted_imp[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"{model_label} — Feature Importance ({task.capitalize()})", pad=10)
    ax.set_xlim(0, sorted_imp.max() * 1.18)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / model_key / f"feature_importance_{task}.png")


def plot_prophet(features_df: pd.DataFrame):
    """Prophet trend + seasonality decomposition for Fresno."""
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    except ImportError:
        print("  Prophet not available, skipping.")
        return

    df = features_df[features_df["county"] == "Fresno"][["date", "aqi_mean"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date": "ds", "aqi_mean": "y"}).sort_values("ds")
    train = df[df["ds"] < "2024-01-01"]
    actual_2024 = df[df["ds"] >= "2024-01-01"]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive",
                    changepoint_prior_scale=0.05)
    model.fit(train)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # ── Plot 1: Forecast vs actual ──
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.18, color="#64B5CD", label="95% uncertainty")
    ax.plot(forecast["ds"], forecast["yhat"], color="#64B5CD", linewidth=1.5, label="Prophet forecast")
    ax.scatter(train["ds"], train["y"], s=2, color="#555555", alpha=0.35, label="Actual (train)")
    ax.scatter(actual_2024["ds"], actual_2024["y"], s=8, color="#E63946",
               alpha=0.8, zorder=5, label="Actual 2024 (holdout)")
    ax.axhline(EXCEEDANCE_THRESHOLD, color="#E63946", linewidth=1.2,
               linestyle="--", alpha=0.7, label=f"AQI {EXCEEDANCE_THRESHOLD} threshold")
    ax.set_title("Fresno — Prophet AQI Forecast vs Actual (2018–2024)", pad=10)
    ax.set_ylabel("AQI")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "prophet" / "forecast.png")

    # ── Plot 2: Trend + seasonality decomposition ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))

    axes[0].plot(forecast["ds"], forecast["trend"], color="#64B5CD", linewidth=1.8)
    axes[0].fill_between(forecast["ds"], forecast["trend_lower"], forecast["trend_upper"],
                         alpha=0.2, color="#64B5CD")
    axes[0].set_title("Long-Term AQI Trend — Fresno", pad=8)
    axes[0].set_ylabel("AQI (trend)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].xaxis.set_major_locator(mdates.YearLocator())

    seasonal = forecast[["ds", "yearly"]].copy()
    seasonal["md"] = seasonal["ds"].dt.strftime("%m-%d")
    smean = seasonal.groupby("md")["yearly"].mean().reset_index().sort_values("md")
    smean["date"] = pd.to_datetime("2023-" + smean["md"], errors="coerce")
    smean = smean.dropna()

    axes[1].fill_between(smean["date"], smean["yearly"], alpha=0.3, color="#C44E52")
    axes[1].plot(smean["date"], smean["yearly"], color="#C44E52", linewidth=1.8)
    axes[1].axhline(0, color="#999999", linewidth=0.9, linestyle="--")

    # Annotate the wildfire season peak
    peak_idx = smean["yearly"].idxmax()
    peak_date = smean.loc[peak_idx, "date"]
    peak_val  = smean.loc[peak_idx, "yearly"]
    axes[1].annotate("Wildfire season\npeak (Aug–Oct)",
                     xy=(peak_date, peak_val),
                     xytext=(pd.Timestamp("2023-05-01"), peak_val * 0.7),
                     arrowprops=dict(arrowstyle="->", color="#555"),
                     fontsize=9, color="#555")

    axes[1].set_title("Yearly Seasonality Component — Fresno", pad=8)
    axes[1].set_ylabel("AQI (seasonal effect)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())

    fig.suptitle("Prophet Decomposition — Fresno AQI", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "prophet" / "seasonality.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _style()

    features_df  = pd.read_csv(FEATURES_PATH)
    metrics      = pd.read_csv(METRICS_PATH)
    cls_preds    = pd.read_csv(CLS_PREDS_PATH)
    reg_preds    = pd.read_csv(REG_PREDS_PATH)

    excluded = {"county", "date", "target_next_day_aqi", "target_next_day_exceedance"}
    feature_cols = [c for c in features_df.columns
                    if c not in excluded and pd.api.types.is_numeric_dtype(features_df[c])]

    print("\n--- Data overview plots ---")
    plot_aqi_timeseries(features_df)
    plot_county_heatmap(features_df)
    plot_exceedance_by_year(features_df)
    plot_correlation_matrix(features_df)

    print("\n--- Model comparison plots ---")
    plot_regression_comparison(metrics)
    plot_classification_comparison(metrics)
    plot_roc_all(cls_preds)
    plot_residuals_all(reg_preds)

    print("\n--- Per-model: predicted vs actual (regression) ---")
    reg_model_map = {
        "ridge":         ("Ridge", "pred_aqi_ridge"),
        "random_forest": ("Random Forest", "prediction_next_day_aqi"),
        "xgboost":       ("XGBoost", "pred_aqi_xgboost"),
        "lightgbm":      ("LightGBM", "pred_aqi_lightgbm"),
        "persistence":   ("Persistence", "pred_aqi_persistence"),
    }
    for key, (label, col) in reg_model_map.items():
        if col in reg_preds.columns:
            plot_predicted_vs_actual(reg_preds, key, label, col)

    print("\n--- Per-model: confusion matrices (classification) ---")
    cls_model_map = {
        "logistic":      ("Logistic Regression", "pred_exc_logistic"),
        "random_forest": ("Random Forest", "prediction_next_day_exceedance"),
        "xgboost":       ("XGBoost", "pred_exc_xgboost"),
        "lightgbm":      ("LightGBM", "pred_exc_lightgbm"),
        "persistence":   ("Persistence", "pred_exc_persistence"),
    }
    for key, (label, col) in cls_model_map.items():
        if col in cls_preds.columns:
            plot_confusion_matrix_model(cls_preds, key, label, col)

    print("\n--- Per-model: feature importance (tree models) ---")
    tree_models = {
        "random_forest": ("Random Forest", "regression_model.joblib", "classification_model.joblib"),
        "xgboost":       ("XGBoost", "xgboost_regression_model.joblib", "xgboost_classification_model.joblib"),
        "lightgbm":      ("LightGBM", "lightgbm_regression_model.joblib", "lightgbm_classification_model.joblib"),
    }
    for key, (label, reg_file, cls_file) in tree_models.items():
        plot_feature_importance_model(MODELS_DIR / reg_file, feature_cols, key, label, "regression")
        plot_feature_importance_model(MODELS_DIR / cls_file, feature_cols, key, label, "classification")

    print("\n--- Prophet plots ---")
    plot_prophet(features_df)

    print("\nDone. All visualizations complete.")
    print("Output: " + str(PLOTS_ROOT.relative_to(PROJECT_ROOT)) + "/")


if __name__ == "__main__":
    main()
