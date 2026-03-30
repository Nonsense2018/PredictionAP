"""
Visualization suite for SJV AQI / School Closure Risk project.
Generates publication-quality plots for the AP Research paper.

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
import matplotlib.patches as mpatches
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
METRICS_PATH   = PROJECT_ROOT / "results" / "models" / "metrics.csv"
CLS_PREDS_PATH = PROJECT_ROOT / "results" / "models" / "classification_predictions.csv"
REG_PREDS_PATH = PROJECT_ROOT / "results" / "models" / "regression_predictions.csv"
MODELS_DIR     = PROJECT_ROOT / "results" / "models"
PLOTS_ROOT     = PROJECT_ROOT / "results" / "plots"

EXCEEDANCE_THRESHOLD = 150  # EPA "Unhealthy" boundary = school closure proxy

# ── Design system ──────────────────────────────────────────────────────────────
COUNTY_COLORS = {
    "Fresno":      "#E63946",
    "Kern":        "#F4A261",
    "Kings":       "#2A9D8F",
    "Madera":      "#457B9D",
    "Merced":      "#1D6FA4",
    "San Joaquin": "#7B2D8B",
    "Stanislaus":  "#E07B39",
    "Tulare":      "#1A7A5E",
}

MODEL_COLORS = {
    "random_forest": "#2196F3",
    "ridge":         "#4CAF50",
    "logistic":      "#FF9800",
    "persistence":   "#9E9E9E",
    "xgboost":       "#E53935",
    "lightgbm":      "#8E24AA",
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "ridge":         "Ridge",
    "logistic":      "Logistic",
    "persistence":   "Persistence",
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
}

SEASON_COLORS = {
    "Winter": "#4FC3F7",
    "Spring": "#81C784",
    "Summer": "#FFB74D",
    "Fall":   "#CE93D8",
}


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#BBBBBB",
        "axes.linewidth":     0.9,
        "axes.grid":          True,
        "grid.color":         "#EEEEEE",
        "grid.linewidth":     0.7,
        "font.family":        "sans-serif",
        "figure.titlesize":   16,
        "figure.titleweight": "bold",
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
        "axes.labelsize":     12,
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#CCCCCC",
        "legend.borderpad":   0.6,
    })


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path.relative_to(PROJECT_ROOT)}")


def _season(month: int) -> str:
    return {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Fall",   10: "Fall",  11: "Fall"}[month]


# ══════════════════════════════════════════════════════════════════════════════
# 1. AQI TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def plot_aqi_timeseries(df: pd.DataFrame):
    """Daily AQI + 30-day rolling mean per county, threshold reference line."""
    counties = sorted(df["county"].unique())
    ncols = 2
    nrows = (len(counties) + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.0), sharex=True)
    axes = axes.flatten()

    for i, county in enumerate(counties):
        ax = axes[i]
        cdf = df[df["county"] == county].copy()
        cdf["date"] = pd.to_datetime(cdf["date"])
        cdf = cdf.sort_values("date")

        color = COUNTY_COLORS.get(county, "#555555")
        ax.fill_between(cdf["date"], cdf["aqi_mean"], alpha=0.12, color=color)
        ax.plot(cdf["date"], cdf["aqi_mean"], lw=0.6, color=color, alpha=0.45, label="Daily AQI")
        roll30 = cdf.set_index("date")["aqi_mean"].rolling("30D", center=True).mean()
        ax.plot(roll30.index, roll30.values, lw=2.2, color=color, label="30-day avg")
        ax.axhline(EXCEEDANCE_THRESHOLD, color="#E63946", lw=1.3,
                   ls="--", alpha=0.75, label=f"Closure threshold ({EXCEEDANCE_THRESHOLD})")
        ax.axhline(100, color="#FF8F00", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(county, fontsize=13, fontweight="bold", pad=6)
        ax.set_ylabel("AQI", fontsize=11)
        ax.set_ylim(0, None)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        if i == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    for j in range(len(counties), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Daily AQI by County — San Joaquin Valley (2018–2024)", y=1.02)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "aqi_timeseries.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. COUNTY × MONTH HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_county_heatmap(df: pd.DataFrame):
    """Mean AQI by county and calendar month, all years pooled."""
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2["month"] = df2["date"].dt.month
    pivot = df2.pivot_table(index="county", columns="month",
                            values="aqi_mean", aggfunc="mean")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.sort_values("Aug", ascending=False)

    fig, ax = plt.subplots(figsize=(13, max(4, len(pivot) * 0.85)))
    im = sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
        linewidths=0.5, linecolor="#DDDDDD",
        annot_kws={"size": 11, "weight": "normal"},
        cbar_kws={"label": "Mean AQI", "shrink": 0.75, "pad": 0.02},
    )
    ax.set_title("Mean AQI by County and Month (2018–2024 Combined)", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.tick_params(axis="y", rotation=0, labelsize=11)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "county_month_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXCEEDANCE DAYS BY YEAR
# ══════════════════════════════════════════════════════════════════════════════

def plot_exceedance_by_year(df: pd.DataFrame):
    """Stacked bar: days with AQI ≥ threshold per county per year."""
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2["year"] = df2["date"].dt.year

    # Count days where ACTUAL AQI (aqi_mean) exceeds threshold
    exc = df2[df2["aqi_mean"] >= EXCEEDANCE_THRESHOLD]
    pivot = exc.groupby(["year", "county"]).size().unstack(fill_value=0)

    counties = pivot.columns.tolist()
    colors = [COUNTY_COLORS.get(c, "#999") for c in counties]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors,
               width=0.62, edgecolor="white", linewidth=0.7)

    ax.set_title(f"Days with AQI ≥ {EXCEEDANCE_THRESHOLD} (School Closure Threshold) by Year",
                 pad=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Exceedance Days", fontsize=12)
    ax.tick_params(axis="x", rotation=0, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    # Annotate total on top of each bar
    for i, year in enumerate(pivot.index):
        total = pivot.loc[year].sum()
        ax.text(i, total + 0.5, str(int(total)), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333")

    # Annotate 2020 as wildfire year
    years = list(pivot.index)
    if 2020 in years:
        idx = years.index(2020)
        ax.annotate("2020 wildfire\nseason", xy=(idx, pivot.loc[2020].sum()),
                    xytext=(idx + 0.7, pivot.loc[2020].sum() + 8),
                    fontsize=9, color="#C62828",
                    arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2))

    ax.legend(title="County", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=10, title_fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "exceedance_by_year.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEASONAL AQI DISTRIBUTION (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def plot_seasonal_distribution(df: pd.DataFrame):
    """Violin plot of AQI distribution by season — shows inversion vs wildfire patterns."""
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2["season"] = df2["date"].dt.month.map(_season)

    season_order = ["Winter", "Spring", "Summer", "Fall"]
    palette = [SEASON_COLORS[s] for s in season_order]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.violinplot(
        data=df2, x="season", y="aqi_mean", order=season_order,
        palette=palette, ax=ax,
        inner="quartile", cut=0, linewidth=1.2, saturation=0.85,
    )

    ax.axhline(EXCEEDANCE_THRESHOLD, color="#E63946", lw=1.5, ls="--",
               alpha=0.8, label=f"Closure threshold (AQI {EXCEEDANCE_THRESHOLD})")
    ax.axhline(100, color="#FF8F00", lw=1.0, ls=":", alpha=0.6,
               label="Unhealthy for sensitive groups (AQI 100)")
    ax.set_title("AQI Distribution by Season — San Joaquin Valley (2018–2024)", pad=12)
    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Daily Mean AQI", fontsize=12)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="upper right")

    # Add season annotations
    season_notes = {
        "Winter": "Temperature\ninversions",
        "Summer": "Wildfire\nsmoke",
    }
    for i, season in enumerate(season_order):
        if season in season_notes:
            ax.text(i, ax.get_ylim()[1] * 0.96, season_notes[season],
                    ha="center", va="top", fontsize=9, color="#555555",
                    style="italic")

    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "seasonal_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance():
    """Horizontal bar chart of RF feature importances for both tasks."""
    FEATURE_LABELS = {
        "aqi_mean":              "AQI (today)",
        "aqi_max":               "AQI max (today)",
        "aqi_lag_1":             "AQI lag 1d",
        "aqi_lag_2":             "AQI lag 2d",
        "aqi_lag_3":             "AQI lag 3d",
        "aqi_roll3_mean":        "AQI rolling 3d mean",
        "aqi_roll7_mean":        "AQI rolling 7d mean",
        "temperature_2m_mean":   "Temperature mean",
        "temperature_2m_max":    "Temperature max",
        "temperature_2m_min":    "Temperature min",
        "precipitation_sum":     "Precipitation",
        "wind_speed_10m_max":    "Wind speed max",
        "min_fire_distance_km":  "Fire distance (km)",
        "fire_event_count_radius": "Fire event count",
        "smoke_present":         "Smoke present",
        "observation_count":     "Observation count",
        "fire_radius_km":        "Fire search radius",
    }

    tasks = [
        ("regression_model.joblib",       "Regression\n(Next-Day AQI)",       "#2196F3"),
        ("classification_model.joblib",   "Classification\n(School Closure Risk)", "#E53935"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, (fname, title, color) in zip(axes, tasks):
        path = MODELS_DIR / fname
        if not path.exists():
            ax.set_visible(False)
            continue
        model = joblib.load(path)
        importances = model.feature_importances_
        names = [FEATURE_LABELS.get(f, f) for f in model.feature_names_in_]

        idx = np.argsort(importances)
        top_n = min(14, len(idx))
        idx = idx[-top_n:]

        bars = ax.barh(
            [names[i] for i in idx],
            importances[idx],
            color=color, alpha=0.82, edgecolor="white", linewidth=0.5,
        )

        # Value labels
        for bar, val in zip(bars, importances[idx]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=9, color="#444444")

        ax.set_title(title, pad=10, fontsize=13)
        ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.set_xlim(0, importances[idx].max() * 1.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", lw=0.6)
        ax.grid(axis="y", visible=False)

    fig.suptitle("Random Forest Feature Importances", y=1.02)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL COMPARISON — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def plot_regression_comparison(metrics: pd.DataFrame):
    """Clean bar chart comparing MAE across regression models, test set."""
    reg = metrics[(metrics["task"] == "regression") & (metrics["split"] == "test")].copy()
    if reg.empty:
        print("  Skipping regression comparison — no data.")
        return

    reg["label"] = reg["model"].map(MODEL_LABELS).fillna(reg["model"])
    reg = reg.sort_values("mae")

    colors = [MODEL_COLORS.get(m, "#999999") for m in reg["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(reg["label"], reg["mae"], color=colors, edgecolor="white",
                  linewidth=0.8, width=0.55, alpha=0.88)

    # Value labels on bars
    for bar, (_, row) in zip(bars, reg.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{row['mae']:.1f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333333")
        if pd.notna(row.get("r2")):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    f"R²={row['r2']:.2f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")

    ax.set_ylabel("Mean Absolute Error (AQI units)", fontsize=12)
    ax.set_title("Regression Performance — Test Set (2024)\nLower MAE = Better", pad=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, reg["mae"].max() * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.7)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "model_comparison_regression.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL COMPARISON — CLASSIFICATION (RECALL FOCUSED)
# ══════════════════════════════════════════════════════════════════════════════

def plot_classification_comparison(metrics: pd.DataFrame):
    """Grouped bar: Precision, Recall, F1 per model — Recall highlighted."""
    cls = metrics[(metrics["task"] == "classification") & (metrics["split"] == "test")].copy()
    if cls.empty:
        print("  Skipping classification comparison — no data.")
        return

    cls["label"] = cls["model"].map(MODEL_LABELS).fillna(cls["model"])
    cls = cls.sort_values("recall", ascending=False)

    metric_map = [
        ("precision", "Precision", "#90CAF9"),
        ("recall",    "Recall",    "#E53935"),   # highlighted
        ("f1",        "F1 Score",  "#A5D6A7"),
    ]

    x = np.arange(len(cls))
    w = 0.24
    offsets = [-1, 0, 1]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for offset, (col, label, color) in zip(offsets, metric_map):
        vals = cls[col].fillna(0).values
        bars = ax.bar(x + offset * w, vals, w, label=label,
                      color=color, edgecolor="white", linewidth=0.7,
                      alpha=0.9 if col == "recall" else 0.75)
        for bar, v in zip(bars, vals):
            if v > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(cls["label"], fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Classification Performance — Test Set (2024)\n"
        f"Predicting School Closure Risk (AQI ≥ {EXCEEDANCE_THRESHOLD})",
        pad=12,
    )
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", lw=0.7)
    ax.grid(axis="x", visible=False)

    # Recall emphasis annotation
    ax.annotate("← Recall prioritized\n   (missing a closure\n   is worse than a\n   false alarm)",
                xy=(0, 0.02), xycoords="axes fraction",
                fontsize=9, color="#C62828", style="italic",
                va="bottom")

    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "model_comparison_classification.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. ROC CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(cls_preds: pd.DataFrame):
    """ROC curves for all classifiers that output probabilities."""
    actual = cls_preds["target_next_day_exceedance"].astype(int)
    if actual.nunique() < 2:
        print("  Skipping ROC: only one class in test set.")
        return

    prob_map = {
        "random_forest": "prediction_probability_exceedance",
        "logistic":      "prob_exc_logistic",
        "xgboost":       "prob_exc_xgboost",
        "lightgbm":      "prob_exc_lightgbm",
    }

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.5, label="Random classifier (AUC = 0.50)")

    plotted = False
    for key, col in prob_map.items():
        if col not in cls_preds.columns:
            continue
        fpr, tpr, _ = roc_curve(actual, cls_preds[col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=MODEL_COLORS.get(key, "#999"),
                label=f"{MODEL_LABELS.get(key, key)}  (AUC = {roc_auc:.3f})")
        plotted = True

    if not plotted:
        print("  Skipping ROC: no probability columns found.")
        plt.close(fig)
        return

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title(
        f"ROC Curves — All Classifiers (Test Set 2024)\n"
        f"School Closure Threshold: AQI ≥ {EXCEEDANCE_THRESHOLD}",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CONFUSION MATRIX — BEST CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cls_preds: pd.DataFrame, metrics: pd.DataFrame):
    """Side-by-side normalized confusion matrices for all available classifiers."""
    actual = cls_preds["target_next_day_exceedance"].astype(int)

    pred_map = {
        "Logistic":      "pred_exc_logistic",
        "Random Forest": "prediction_next_day_exceedance",
        "Persistence":   "pred_exc_persistence",
        "XGBoost":       "pred_exc_xgboost",
        "LightGBM":      "pred_exc_lightgbm",
    }
    available = [(name, col) for name, col in pred_map.items()
                 if col in cls_preds.columns]

    if not available:
        print("  Skipping confusion matrix — no prediction columns found.")
        return

    ncols = min(3, len(available))
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if len(available) > 1 else [axes]

    labels_short = [f"AQI < {EXCEEDANCE_THRESHOLD}\n(Safe)", f"AQI ≥ {EXCEEDANCE_THRESHOLD}\n(Closure)"]

    for ax, (name, col) in zip(axes, available):
        pred = cls_preds[col].astype(int)
        cm = confusion_matrix(actual, pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        # Custom annotated heatmap (no colorbar)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j]
                count = cm[i, j]
                txt_color = "white" if pct > 0.55 else "#333333"
                ax.text(j, i, f"{pct:.0%}\n({count})",
                        ha="center", va="center", fontsize=12,
                        color=txt_color, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_short, fontsize=10)
        ax.set_yticklabels(labels_short, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11, labelpad=8)
        ax.set_ylabel("Actual", fontsize=11, labelpad=8)

        rec  = recall_score(actual, pred, zero_division=0)
        prec = precision_score(actual, pred, zero_division=0)
        f1   = f1_score(actual, pred, zero_division=0)
        ax.set_title(f"{name}\nPrec={prec:.2f}  Rec={rec:.2f}  F1={f1:.2f}",
                     fontsize=12, pad=10)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Confusion Matrices — Test Set (2024)\nSchool Closure Proxy: AQI ≥ {EXCEEDANCE_THRESHOLD}",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "confusion_matrices.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. PREDICTED VS ACTUAL — BEST REGRESSION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(reg_preds: pd.DataFrame):
    """Predicted vs actual scatter for Random Forest, colored by county."""
    pred_col = "prediction_next_day_aqi"
    if pred_col not in reg_preds.columns:
        print("  Skipping predicted vs actual — column missing.")
        return

    sub = reg_preds[["county", "target_next_day_aqi", pred_col]].dropna()
    # Clip to valid AQI range
    sub = sub[(sub["target_next_day_aqi"] >= 0) & (sub["target_next_day_aqi"] <= 400)]
    sub = sub[(sub[pred_col] >= 0) & (sub[pred_col] <= 400)]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 7))

    for county, group in sub.groupby("county"):
        ax.scatter(group["target_next_day_aqi"], group[pred_col],
                   s=14, alpha=0.5, color=COUNTY_COLORS.get(county, "#999"),
                   label=county, edgecolors="none")

    # Perfect prediction line
    lo = max(0, min(sub["target_next_day_aqi"].min(), sub[pred_col].min()) - 5)
    hi = max(sub["target_next_day_aqi"].max(), sub[pred_col].max()) + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.55, label="Perfect prediction", zorder=5)

    # Threshold reference
    ax.axvline(EXCEEDANCE_THRESHOLD, color="#E63946", lw=1.0, ls=":", alpha=0.6)
    ax.axhline(EXCEEDANCE_THRESHOLD, color="#E63946", lw=1.0, ls=":", alpha=0.6,
               label=f"Closure threshold (AQI {EXCEEDANCE_THRESHOLD})")

    mae = (sub[pred_col] - sub["target_next_day_aqi"]).abs().mean()
    r2  = 1 - ((sub[pred_col] - sub["target_next_day_aqi"])**2).sum() / \
              ((sub["target_next_day_aqi"] - sub["target_next_day_aqi"].mean())**2).sum()

    ax.set_xlabel("Actual Next-Day AQI", fontsize=12)
    ax.set_ylabel("Predicted Next-Day AQI", fontsize=12)
    ax.set_title(f"Random Forest — Predicted vs Actual (Test Set 2024)\nMAE = {mae:.1f}  |  R² = {r2:.3f}",
                 pad=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", fontsize=9, markerscale=1.5)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "predicted_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _apply_style()

    print("\nLoading data …")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing: {FEATURES_PATH}")
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])

    metrics = pd.DataFrame()
    if METRICS_PATH.exists():
        raw = pd.read_csv(METRICS_PATH)
        # Handle both wide and long format
        if "task" not in raw.columns:
            rows = []
            for _, r in raw.iterrows():
                rows.append({"split": r["split"], "task": "regression", "model": r.get("model", "random_forest"),
                              "mae": r.get("mae"), "rmse": r.get("rmse"), "r2": r.get("r2")})
                rows.append({"split": r["split"], "task": "classification", "model": r.get("model", "random_forest"),
                              "precision": r.get("precision"), "recall": r.get("recall"),
                              "f1": r.get("f1"), "roc_auc": r.get("roc_auc")})
            metrics = pd.DataFrame(rows)
        else:
            metrics = raw

    reg_preds = pd.DataFrame()
    if REG_PREDS_PATH.exists():
        reg_preds = pd.read_csv(REG_PREDS_PATH)
        reg_preds["date"] = pd.to_datetime(reg_preds["date"])

    cls_preds = pd.DataFrame()
    if CLS_PREDS_PATH.exists():
        cls_preds = pd.read_csv(CLS_PREDS_PATH)
        cls_preds["date"] = pd.to_datetime(cls_preds["date"])

    print("\nGenerating plots …")

    print("\n[1/10] AQI time series")
    plot_aqi_timeseries(df)

    print("[2/10] County × month heatmap")
    plot_county_heatmap(df)

    print("[3/10] Exceedance days by year")
    plot_exceedance_by_year(df)

    print("[4/10] Seasonal AQI distribution")
    plot_seasonal_distribution(df)

    print("[5/10] Feature importance")
    plot_feature_importance()

    if not metrics.empty:
        print("[6/10] Regression model comparison")
        plot_regression_comparison(metrics)

        print("[7/10] Classification model comparison")
        plot_classification_comparison(metrics)
    else:
        print("[6/10] Skipping regression comparison — no metrics.csv")
        print("[7/10] Skipping classification comparison — no metrics.csv")

    if not cls_preds.empty:
        print("[8/10] ROC curves")
        plot_roc_curves(cls_preds)

        print("[9/10] Confusion matrices")
        plot_confusion_matrix(cls_preds, metrics)
    else:
        print("[8/10] Skipping ROC — no classification_predictions.csv")
        print("[9/10] Skipping confusion matrix — no classification_predictions.csv")

    if not reg_preds.empty:
        print("[10/10] Predicted vs actual")
        plot_predicted_vs_actual(reg_preds)
    else:
        print("[10/10] Skipping predicted vs actual — no regression_predictions.csv")

    print(f"\nAll plots saved to: results/plots/")


if __name__ == "__main__":
    main()
