"""
Visualization suite — SJV AQI Risk / School Hazardous Event Prediction
Generates 15 publication-quality plots for the AP Research paper.

Usage:
    python scripts/visualize/run_all_visualizations.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
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

EXCEEDANCE_THRESHOLD = 100   # AQI ≥ 100 → school hazardous event proxy
TRAIN_END  = pd.Timestamp("2022-12-31")
VAL_END    = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END   = pd.Timestamp("2024-12-31")

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
    "random_forest": "#2563EB",
    "ridge":         "#16A34A",
    "logistic":      "#D97706",
    "persistence":   "#6B7280",
    "xgboost":       "#DC2626",
    "lightgbm":      "#7C3AED",
}

MODEL_LABELS = {
    "random_forest": "Random Forest",
    "ridge":         "Ridge",
    "logistic":      "Logistic",
    "persistence":   "Persistence",
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
}

FEATURE_LABELS = {
    "aqi_mean":               "AQI — same day (mean)",
    "aqi_max":                "AQI — same day (max)",
    "aqi_lag_1":              "AQI — 1 day ago",
    "aqi_lag_2":              "AQI — 2 days ago",
    "aqi_lag_3":              "AQI — 3 days ago",
    "aqi_roll3_mean":         "AQI — 3-day rolling mean",
    "aqi_roll7_mean":         "AQI — 7-day rolling mean",
    "temperature_2m_mean":    "Temperature mean (°C)",
    "temperature_2m_max":     "Temperature max (°C)",
    "temperature_2m_min":     "Temperature min (°C)",
    "precipitation_sum":      "Precipitation (mm)",
    "wind_speed_10m_max":     "Wind speed max (km/h)",
    "fire_event_count_radius":"Wildfire events within 150 km",
    "smoke_present":          "Smoke present (binary)",
    "min_fire_distance_km":   "Distance to nearest fire (km)",
    "observation_count":      "Monitor readings (count)",
}

SEASON_ORDER  = ["Winter", "Spring", "Summer", "Fall"]
SEASON_COLORS = {"Winter": "#60A5FA", "Spring": "#4ADE80",
                 "Summer": "#FB923C", "Fall":   "#C084FC"}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _apply_style():
    """Global rcParams — applied once at startup."""
    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "#FAFAFA",
        "axes.edgecolor":      "#D1D5DB",
        "axes.linewidth":      0.8,
        "axes.grid":           True,
        "grid.color":          "#E5E7EB",
        "grid.linewidth":      0.6,
        "grid.alpha":          1.0,
        "font.family":         "sans-serif",
        "font.size":           11,
        "figure.titlesize":    14,
        "figure.titleweight":  "bold",
        "axes.titlesize":      12,
        "axes.titleweight":    "bold",
        "axes.titlepad":       10,
        "axes.labelsize":      11,
        "axes.labelpad":       6,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "xtick.major.pad":     4,
        "ytick.major.pad":     4,
        "legend.fontsize":     10,
        "legend.title_fontsize": 10,
        "legend.framealpha":   0.95,
        "legend.edgecolor":    "#D1D5DB",
        "legend.borderpad":    0.5,
        "legend.handlelength": 1.5,
        "lines.linewidth":     1.8,
        "patch.linewidth":     0.5,
    })


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(PROJECT_ROOT)}")


def _season(month: int) -> str:
    return {12: "Winter", 1: "Winter",  2: "Winter",
             3: "Spring", 4: "Spring",  5: "Spring",
             6: "Summer", 7: "Summer",  8: "Summer",
             9: "Fall",  10: "Fall",   11: "Fall"}[month]


def _clean_spines(ax, keep=("left", "bottom")):
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(spine in keep)


# ══════════════════════════════════════════════════════════════════════════════
# 1. AQI DISTRIBUTION BY COUNTY
# ══════════════════════════════════════════════════════════════════════════════

def plot_aqi_distribution(df: pd.DataFrame):
    """Violin + box plot of AQI distribution per county, ordered by median."""
    order = (df.groupby("county")["aqi_mean"].median()
               .sort_values(ascending=False).index.tolist())
    palette = [COUNTY_COLORS.get(c, "#999999") for c in order]

    fig, ax = plt.subplots(figsize=(13, 6))

    sns.violinplot(data=df, x="county", y="aqi_mean", order=order,
                   palette=palette, ax=ax,
                   inner=None, cut=0, linewidth=0.8,
                   saturation=0.80, alpha=0.65)
    sns.boxplot(data=df, x="county", y="aqi_mean", order=order,
                ax=ax, width=0.14, fliersize=1.5,
                boxprops=dict(facecolor="white", linewidth=1.0),
                medianprops=dict(color="#111827", linewidth=2.0),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                flierprops=dict(marker="o", markeredgewidth=0.4,
                                alpha=0.3, markersize=2.5))

    ax.axhline(EXCEEDANCE_THRESHOLD, color="#DC2626", lw=1.4, ls="--",
               alpha=0.85, label=f"Hazardous event threshold (AQI {EXCEEDANCE_THRESHOLD})")
    ax.set_title("AQI Distribution by County — San Joaquin Valley, 2018–2024")
    ax.set_xlabel("County")
    ax.set_ylabel("Daily Mean AQI")
    ax.set_ylim(0, None)
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(loc="upper right")
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "aqi_distribution_by_county.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. COUNTY × MONTH HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_county_heatmap(df: pd.DataFrame):
    """Mean AQI by county and calendar month, all years pooled."""
    df2 = df.copy()
    df2["month"] = df2["date"].dt.month
    pivot = df2.pivot_table(index="county", columns="month",
                            values="aqi_mean", aggfunc="mean")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = pivot.sort_values("Aug", ascending=False)

    fig, ax = plt.subplots(figsize=(13, max(4, len(pivot) * 0.9)))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.4, linecolor="#E5E7EB",
                annot_kws={"size": 10, "weight": "normal"},
                cbar_kws={"label": "Mean AQI", "shrink": 0.72, "pad": 0.02})
    ax.set_title("Mean AQI by County and Month, 2018–2024 Combined")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    ax.collections[0].colorbar.ax.tick_params(labelsize=9)
    ax.collections[0].colorbar.ax.set_ylabel("Mean AQI", fontsize=10)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "county_month_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCHOOL HAZARDOUS EVENT DAYS BY YEAR
# ══════════════════════════════════════════════════════════════════════════════

def plot_hazardous_days_by_year(df: pd.DataFrame):
    """Stacked bar of county-days with AQI ≥ threshold per year."""
    df2 = df.copy()
    df2["year"] = df2["date"].dt.year
    exc   = df2[df2["aqi_mean"] >= EXCEEDANCE_THRESHOLD]
    pivot = exc.groupby(["year", "county"]).size().unstack(fill_value=0)

    counties = pivot.columns.tolist()
    colors   = [COUNTY_COLORS.get(c, "#999999") for c in counties]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=colors,
               width=0.60, edgecolor="white", linewidth=0.5)

    # Total labels on top
    for i, year in enumerate(pivot.index):
        total = int(pivot.loc[year].sum())
        ax.text(i, total + 0.5, str(total), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1F2937")

    ax.set_title(f"School Hazardous Event Days per Year (AQI ≥ {EXCEEDANCE_THRESHOLD})")
    ax.set_xlabel("Year")
    ax.set_ylabel("County-Days Exceeding Threshold")
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.legend(title="County", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, title_fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "hazardous_days_by_year.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. AQI LAG AUTOCORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_acf(df: pd.DataFrame):
    """Manual ACF of daily AQI across all counties pooled, lags 1–30."""
    series = (df.sort_values(["county", "date"])
                .groupby("county")["aqi_mean"]
                .apply(lambda x: x.dropna().values))

    # Pool all county series (within-county only — no cross-county leakage)
    all_acf = []
    NLAGS = 30
    for county_vals in series:
        v = county_vals - county_vals.mean()
        var = (v ** 2).mean()
        if var == 0:
            continue
        acf_vals = [1.0]
        for lag in range(1, NLAGS + 1):
            cov = (v[lag:] * v[:-lag]).mean()
            acf_vals.append(cov / var)
        all_acf.append(acf_vals)

    mean_acf = np.array(all_acf).mean(axis=0)
    lags = np.arange(0, NLAGS + 1)
    n = int(df["aqi_mean"].notna().sum())
    ci = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Stem plot
    ax.axhline(0, color="#374151", lw=0.8)
    ax.fill_between(lags, ci, -ci, alpha=0.12, color="#2563EB", label="95% confidence interval")
    for lag, val in zip(lags[1:], mean_acf[1:]):
        color = "#DC2626" if abs(val) > ci else "#2563EB"
        ax.plot([lag, lag], [0, val], color=color, lw=1.8, solid_capstyle="round")
        ax.scatter(lag, val, s=28, color=color, zorder=5)

    ax.scatter(0, 1.0, s=28, color="#2563EB", zorder=5)
    ax.plot([0, 0], [0, 1.0], color="#2563EB", lw=1.8)

    ax.axhline(ci,  color="#6B7280", lw=0.9, ls="--", alpha=0.7)
    ax.axhline(-ci, color="#6B7280", lw=0.9, ls="--", alpha=0.7)

    ax.set_title("AQI Autocorrelation by Lag — All Counties Pooled\n"
                 "Red bars exceed the 95% confidence interval")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(-0.5, NLAGS + 0.5)
    ax.set_xticks(range(0, NLAGS + 1, 5))
    ax.legend(loc="upper right", fontsize=10)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "aqi_autocorrelation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. AVERAGED FEATURE IMPORTANCE ACROSS TREE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance():
    """Average feature importance across RF regression + RF classification."""
    paths = [
        MODELS_DIR / "regression_model.joblib",
        MODELS_DIR / "classification_model.joblib",
    ]
    importance_dfs = []
    for p in paths:
        if not p.exists():
            continue
        model = joblib.load(p)
        imp = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        importance_dfs.append(imp)

    if not importance_dfs:
        print("  Skipping feature importance — no model files found.")
        return

    combined = pd.concat(importance_dfs, axis=1).fillna(0)
    avg_imp  = combined.mean(axis=1).sort_values(ascending=True)

    labels  = [FEATURE_LABELS.get(f, f) for f in avg_imp.index]
    colors  = ["#DC2626" if avg_imp[f] == avg_imp.max()
               else "#2563EB" for f in avg_imp.index]

    fig, ax = plt.subplots(figsize=(10, max(5, len(avg_imp) * 0.45)))
    bars = ax.barh(labels, avg_imp.values, color=colors,
                   alpha=0.82, edgecolor="white", linewidth=0.4, height=0.65)

    for bar, val in zip(bars, avg_imp.values):
        ax.text(bar.get_width() + avg_imp.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                fontsize=9, color="#374151")

    ax.set_title("Feature Importance — Averaged Across Tree Models\n"
                 "(Random Forest Regression + Classification)")
    ax.set_xlabel("Mean Decrease in Impurity (averaged)")
    ax.set_xlim(0, avg_imp.max() * 1.20)
    ax.tick_params(axis="y", labelsize=10)
    _clean_spines(ax)
    ax.grid(axis="x", lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE–TARGET CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_correlation(df: pd.DataFrame):
    """Spearman correlation of each feature with next-day AQI target."""
    excluded = {"county", "date", "target_next_day_exceedance"}
    feature_cols = [c for c in df.columns
                    if c not in excluded
                    and pd.api.types.is_numeric_dtype(df[c])
                    and c != "target_next_day_aqi"
                    and df[c].notna().any()
                    and df[c].nunique() > 1]

    corrs = {}
    for col in feature_cols:
        valid = df[["target_next_day_aqi", col]].dropna()
        if len(valid) < 30:
            continue
        corr = valid["target_next_day_aqi"].corr(valid[col], method="spearman")
        corrs[col] = corr

    corr_series = pd.Series(corrs).sort_values()
    labels = [FEATURE_LABELS.get(f, f) for f in corr_series.index]
    colors = ["#DC2626" if v > 0 else "#2563EB" for v in corr_series.values]

    fig, ax = plt.subplots(figsize=(10, max(5, len(corr_series) * 0.45)))
    bars = ax.barh(labels, corr_series.values, color=colors,
                   alpha=0.82, edgecolor="white", linewidth=0.4, height=0.65)
    ax.axvline(0, color="#374151", lw=0.9)

    for bar, val in zip(bars, corr_series.values):
        x_pos = val + (0.01 if val >= 0 else -0.01)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha,
                fontsize=9, color="#374151")

    ax.set_title("Spearman Correlation of Each Feature with Next-Day AQI\n"
                 "Red = positive association, Blue = negative association")
    ax.set_xlabel("Spearman Correlation Coefficient")
    lim = max(abs(corr_series.values)) * 1.22
    ax.set_xlim(-lim, lim)
    ax.tick_params(axis="y", labelsize=10)
    _clean_spines(ax)
    ax.grid(axis="x", lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "feature_target_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. WILDFIRE EVENTS OVERLAID ON AQI SPIKES
# ══════════════════════════════════════════════════════════════════════════════

def plot_wildfire_aqi_overlay(df: pd.DataFrame):
    """AQI time series for Fresno with wildfire event days highlighted."""
    county = "Fresno"
    cdf = df[df["county"] == county].copy().sort_values("date")

    if cdf.empty or "smoke_present" not in cdf.columns:
        print(f"  Skipping wildfire overlay — no data for {county}.")
        return

    fire_days = cdf[cdf["smoke_present"] == 1]

    fig, ax = plt.subplots(figsize=(14, 5))
    color = COUNTY_COLORS[county]

    ax.fill_between(cdf["date"], cdf["aqi_mean"], alpha=0.10, color=color)
    ax.plot(cdf["date"], cdf["aqi_mean"], lw=0.7, color=color,
            alpha=0.50, label="Daily AQI")
    roll = cdf.set_index("date")["aqi_mean"].rolling("30D", center=True).mean()
    ax.plot(roll.index, roll.values, lw=2.0, color=color, label="30-day rolling mean")

    if not fire_days.empty:
        ax.scatter(fire_days["date"], fire_days["aqi_mean"],
                   s=22, color="#DC2626", zorder=6, alpha=0.75,
                   label=f"Active wildfire within 150 km ({len(fire_days)} days)")

    ax.axhline(EXCEEDANCE_THRESHOLD, color="#DC2626", lw=1.3, ls="--",
               alpha=0.80, label=f"Hazardous event threshold (AQI {EXCEEDANCE_THRESHOLD})")

    ax.set_title(f"Daily AQI with Active Wildfire Days — {county} County, 2018–2024")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Mean AQI")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=9)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "wildfire_aqi_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. AQI vs METEOROLOGICAL VARIABLES
# ══════════════════════════════════════════════════════════════════════════════

def plot_met_scatter(df: pd.DataFrame):
    """4-panel scatter: AQI vs temperature, wind, precipitation, fire distance."""
    panels = [
        ("temperature_2m_max",   "Max Temperature (°C)",    "#E07B39"),
        ("wind_speed_10m_max",   "Max Wind Speed (km/h)",   "#1D6FA4"),
        ("precipitation_sum",    "Precipitation (mm)",      "#2A9D8F"),
        ("min_fire_distance_km", "Distance to Nearest Fire (km)", "#DC2626"),
    ]
    panels = [(col, lbl, c) for col, lbl, c in panels if col in df.columns]
    if not panels:
        print("  Skipping met scatter — no meteorological columns found.")
        return

    ncols = 2
    nrows = (len(panels) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 4.5))
    axes = np.array(axes).flatten()

    for ax, (col, xlabel, color) in zip(axes, panels):
        sub = df[["aqi_mean", col]].dropna()
        # Cap sentinel fire distance for readability
        if col == "min_fire_distance_km":
            sub = sub[sub[col] < 1000]

        ax.scatter(sub[col], sub["aqi_mean"], s=4, alpha=0.25,
                   color=color, edgecolors="none", rasterized=True)

        # Trend line
        try:
            m, b = np.polyfit(sub[col], sub["aqi_mean"], 1)
            x_line = np.linspace(sub[col].min(), sub[col].max(), 200)
            ax.plot(x_line, m * x_line + b, color="#1F2937",
                    lw=1.8, alpha=0.85, label=f"Trend (slope={m:.2f})")
        except Exception:
            pass

        corr = sub["aqi_mean"].corr(sub[col], method="spearman")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Daily Mean AQI")
        ax.set_title(f"AQI vs {xlabel}\nSpearman r = {corr:+.3f}")
        ax.axhline(EXCEEDANCE_THRESHOLD, color="#DC2626", lw=1.0,
                   ls="--", alpha=0.60)
        ax.legend(fontsize=9, loc="upper right")
        _clean_spines(ax)

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Next-Day AQI vs Meteorological Predictors — All Counties, 2018–2024",
                 y=1.01)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "aqi_vs_met_variables.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. REGRESSION PERFORMANCE — ALL MODELS, VAL + TEST
# ══════════════════════════════════════════════════════════════════════════════

def plot_regression_comparison(metrics: pd.DataFrame):
    """Grouped bar: MAE on validation and test for every regression model."""
    reg = metrics[metrics["task"] == "regression"].copy()
    if reg.empty:
        print("  Skipping regression comparison — no data.")
        return

    reg["label"] = reg["model"].map(MODEL_LABELS).fillna(reg["model"])
    order = (reg[reg["split"] == "test"].sort_values("mae")["model"].tolist())
    order = [m for m in order if m in reg["model"].values]

    splits   = ["validation", "test"]
    x        = np.arange(len(order))
    w        = 0.32
    fig, ax  = plt.subplots(figsize=(11, 5.5))

    for i, split in enumerate(splits):
        sub  = reg[reg["split"] == split].set_index("model")
        vals = [sub.loc[m, "mae"] if m in sub.index else np.nan for m in order]
        r2s  = [sub.loc[m, "r2"]  if m in sub.index else np.nan for m in order]
        cols = [MODEL_COLORS.get(m, "#9CA3AF") for m in order]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w,
                      color=cols, alpha=0.75 if split == "validation" else 0.95,
                      edgecolor="white", linewidth=0.5,
                      label=split.capitalize(),
                      hatch="////" if split == "validation" else "")
        for bar, v, r2 in zip(bars, vals, r2s):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=9, color="#1F2937")
                if not np.isnan(r2):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() / 2,
                            f"R²={r2:.2f}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")

    labels = [MODEL_LABELS.get(m, m) for m in order]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean Absolute Error (AQI units)")
    ax.set_title("Regression Performance — All Models, Validation & Test\n"
                 "Lower MAE is better. R² shown inside bars.")
    ax.legend(title="Split", fontsize=10)
    ax.set_ylim(0, max(reg["mae"].dropna()) * 1.30)
    _clean_spines(ax)
    ax.grid(axis="y", lw=0.6)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "regression_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. CLASSIFICATION PERFORMANCE — ALL MODELS, VAL + TEST
# ══════════════════════════════════════════════════════════════════════════════

def plot_classification_comparison(metrics: pd.DataFrame):
    """Grouped bar: Recall, F1, AUC per model — both splits."""
    cls = metrics[metrics["task"] == "classification"].copy()
    if cls.empty:
        print("  Skipping classification comparison — no data.")
        return

    cls["label"] = cls["model"].map(MODEL_LABELS).fillna(cls["model"])
    order = (cls[cls["split"] == "test"].sort_values("recall", ascending=False)
               ["model"].tolist())
    order = [m for m in order if m in cls["model"].values]

    metric_cfg = [
        ("recall",   "Recall",   "#DC2626"),
        ("f1",       "F1 Score", "#16A34A"),
        ("roc_auc",  "AUC",      "#2563EB"),
    ]

    x   = np.arange(len(order))
    w   = 0.20
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, split in zip(axes, ["validation", "test"]):
        sub = cls[cls["split"] == split].set_index("model")
        for k, (col, label, color) in enumerate(metric_cfg):
            vals = [sub.loc[m, col] if m in sub.index and col in sub.columns
                    else np.nan for m in order]
            offset = (k - 1) * w
            bars = ax.bar(x + offset, [v if not np.isnan(v) else 0 for v in vals],
                          w, label=label, color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if not np.isnan(v) and v > 0.04:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=8, color="#1F2937")

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in order], fontsize=10)
        ax.set_ylim(0, 1.10)
        ax.set_title(f"{split.capitalize()} Set")
        ax.set_ylabel("Score" if split == "validation" else "")
        _clean_spines(ax)
        ax.grid(axis="y", lw=0.6)
        ax.grid(axis="x", visible=False)
        if split == "validation":
            ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(f"Classification Performance — All Models\n"
                 f"School Hazardous Event Threshold: AQI ≥ {EXCEEDANCE_THRESHOLD}  |  Recall is primary metric")
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "classification_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. ROC CURVES — ALL CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(cls_preds: pd.DataFrame):
    """ROC curves for all classifiers overlaid on one plot."""
    actual = cls_preds["target_next_day_exceedance"].astype(int)
    if actual.nunique() < 2:
        print("  Skipping ROC — only one class in test set.")
        return

    prob_cols = {
        "random_forest": "prediction_probability_exceedance",
        "logistic":      "prob_exc_logistic",
        "xgboost":       "prob_exc_xgboost",
        "lightgbm":      "prob_exc_lightgbm",
    }

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.0, ls="--",
            label="No-skill classifier (AUC = 0.50)")

    plotted = False
    for key, col in prob_cols.items():
        if col not in cls_preds.columns:
            continue
        fpr, tpr, _ = roc_curve(actual, cls_preds[col])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.0, color=MODEL_COLORS.get(key, "#6B7280"),
                label=f"{MODEL_LABELS.get(key, key)}  (AUC = {roc_auc:.3f})")
        plotted = True

    if not plotted:
        print("  Skipping ROC — no probability columns found.")
        plt.close(fig)
        return

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curves — All Classifiers, Test Set 2024\n"
                 f"School Hazardous Event Threshold: AQI ≥ {EXCEEDANCE_THRESHOLD}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "roc_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. PRECISION-RECALL CURVES — ALL CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_precision_recall_curves(cls_preds: pd.DataFrame):
    """Precision-recall curves for all classifiers — more informative for imbalanced data."""
    actual = cls_preds["target_next_day_exceedance"].astype(int)
    if actual.nunique() < 2:
        print("  Skipping PR curve — only one class in test set.")
        return

    prevalence = actual.mean()
    prob_cols = {
        "random_forest": "prediction_probability_exceedance",
        "logistic":      "prob_exc_logistic",
        "xgboost":       "prob_exc_xgboost",
        "lightgbm":      "prob_exc_lightgbm",
    }

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.axhline(prevalence, color="#9CA3AF", lw=1.0, ls="--",
               label=f"No-skill baseline (precision = {prevalence:.3f})")

    plotted = False
    for key, col in prob_cols.items():
        if col not in cls_preds.columns:
            continue
        prec, rec, _ = precision_recall_curve(actual, cls_preds[col])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2.0, color=MODEL_COLORS.get(key, "#6B7280"),
                label=f"{MODEL_LABELS.get(key, key)}  (AUC = {pr_auc:.3f})")
        plotted = True

    if not plotted:
        print("  Skipping PR curve — no probability columns found.")
        plt.close(fig)
        return

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curves — All Classifiers, Test Set 2024\n"
                 f"School Hazardous Event Threshold: AQI ≥ {EXCEEDANCE_THRESHOLD}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=10)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "precision_recall_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13. PREDICTED VS ACTUAL — ALL REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(reg_preds: pd.DataFrame):
    """Predicted vs actual scatter for all available regression models."""
    pred_cols = {
        "random_forest": "prediction_next_day_aqi",
        "ridge":         "pred_aqi_ridge",
        "persistence":   "pred_aqi_persistence",
        "xgboost":       "pred_aqi_xgboost",
        "lightgbm":      "pred_aqi_lightgbm",
    }
    available = [(k, v) for k, v in pred_cols.items() if v in reg_preds.columns]
    if not available:
        print("  Skipping predicted vs actual — no prediction columns found.")
        return

    ncols = min(3, len(available))
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    actual = reg_preds["target_next_day_aqi"]

    for ax, (model_key, pred_col) in zip(axes, available):
        sub = reg_preds[["county", "target_next_day_aqi", pred_col]].dropna()
        sub = sub[(sub["target_next_day_aqi"] >= 0) & (sub[pred_col] >= 0)]

        for county, grp in sub.groupby("county"):
            ax.scatter(grp["target_next_day_aqi"], grp[pred_col],
                       s=8, alpha=0.35, color=COUNTY_COLORS.get(county, "#9CA3AF"),
                       edgecolors="none", label=county)

        lo = 0
        hi = max(sub["target_next_day_aqi"].max(), sub[pred_col].max()) + 5
        ax.plot([lo, hi], [lo, hi], color="#374151", lw=1.4, ls="--",
                alpha=0.60, label="Perfect prediction")
        ax.axvline(EXCEEDANCE_THRESHOLD, color="#DC2626", lw=0.9, ls=":", alpha=0.55)
        ax.axhline(EXCEEDANCE_THRESHOLD, color="#DC2626", lw=0.9, ls=":", alpha=0.55)

        mae = (sub[pred_col] - sub["target_next_day_aqi"]).abs().mean()
        r2  = 1 - (((sub[pred_col] - sub["target_next_day_aqi"]) ** 2).sum() /
                   ((sub["target_next_day_aqi"] - sub["target_next_day_aqi"].mean()) ** 2).sum())
        ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)}\nMAE = {mae:.1f}  |  R² = {r2:.3f}")
        ax.set_xlabel("Actual Next-Day AQI")
        ax.set_ylabel("Predicted Next-Day AQI")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        _clean_spines(ax)

    # Shared county legend on last used axis
    if available:
        handles = [plt.scatter([], [], s=20, color=COUNTY_COLORS.get(c, "#999"),
                               label=c) for c in COUNTY_COLORS]
        axes[len(available) - 1].legend(handles=handles, title="County",
                                        fontsize=8, title_fontsize=9,
                                        loc="upper left", markerscale=1.5)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Predicted vs Actual Next-Day AQI — Test Set 2024\n"
                 "Dotted red lines mark the school hazardous event threshold", y=1.01)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "predicted_vs_actual.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14. PREDICTION ERROR BY COUNTY
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_by_county(reg_preds: pd.DataFrame):
    """Box plot of residuals per county — best available regression model."""
    pred_col = next((c for c in ["prediction_next_day_aqi", "pred_aqi_ridge"]
                     if c in reg_preds.columns), None)
    if pred_col is None:
        print("  Skipping error by county — no prediction columns.")
        return

    sub = reg_preds[["county", "target_next_day_aqi", pred_col]].dropna()
    sub["residual"] = sub[pred_col] - sub["target_next_day_aqi"]

    order  = sub.groupby("county")["residual"].median().sort_values().index.tolist()
    palette = [COUNTY_COLORS.get(c, "#9CA3AF") for c in order]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(data=sub, x="county", y="residual", order=order,
                palette=palette, ax=ax, width=0.50, fliersize=2.0,
                flierprops=dict(alpha=0.3, markersize=2.5),
                medianprops=dict(color="#111827", linewidth=2.0))

    ax.axhline(0, color="#374151", lw=1.2, ls="--", alpha=0.80,
               label="Zero error (perfect prediction)")
    ax.set_title("Next-Day AQI Prediction Residuals by County — Test Set 2024\n"
                 "Positive = over-prediction, Negative = under-prediction")
    ax.set_xlabel("County")
    ax.set_ylabel("Residual (Predicted − Actual AQI)")
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(fontsize=10)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "error_by_county.png")


# ══════════════════════════════════════════════════════════════════════════════
# 15. PREDICTION ERROR BY SEASON
# ══════════════════════════════════════════════════════════════════════════════

def plot_error_by_season(reg_preds: pd.DataFrame):
    """Box plot of residuals per season — reveals when the model struggles."""
    pred_col = next((c for c in ["prediction_next_day_aqi", "pred_aqi_ridge"]
                     if c in reg_preds.columns), None)
    if pred_col is None:
        print("  Skipping error by season — no prediction columns.")
        return

    sub = reg_preds[["date", "target_next_day_aqi", pred_col]].dropna().copy()
    sub["date"]     = pd.to_datetime(sub["date"])
    sub["season"]   = sub["date"].dt.month.map(_season)
    sub["residual"] = sub[pred_col] - sub["target_next_day_aqi"]

    palette = [SEASON_COLORS[s] for s in SEASON_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(data=sub, x="season", y="residual", order=SEASON_ORDER,
                palette=palette, ax=ax, width=0.45, fliersize=2.0,
                flierprops=dict(alpha=0.3, markersize=2.5),
                medianprops=dict(color="#111827", linewidth=2.0))

    ax.axhline(0, color="#374151", lw=1.2, ls="--", alpha=0.80,
               label="Zero error (perfect prediction)")

    # Annotate known hard seasons
    for i, season in enumerate(SEASON_ORDER):
        n = (sub["season"] == season).sum()
        ax.text(i, ax.get_ylim()[0] * 0.96, f"n={n}",
                ha="center", va="bottom", fontsize=9, color="#6B7280")

    ax.set_title("Next-Day AQI Prediction Residuals by Season — Test Set 2024\n"
                 "Reveals where the model struggles (inversions in Winter, smoke in Summer/Fall)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Residual (Predicted − Actual AQI)")
    ax.legend(fontsize=10)
    _clean_spines(ax)
    fig.tight_layout()
    _save(fig, PLOTS_ROOT / "error_by_season.png")


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
        metrics = pd.read_csv(METRICS_PATH)

    reg_preds = pd.DataFrame()
    if REG_PREDS_PATH.exists():
        reg_preds = pd.read_csv(REG_PREDS_PATH)
        reg_preds["date"] = pd.to_datetime(reg_preds["date"])

    cls_preds = pd.DataFrame()
    if CLS_PREDS_PATH.exists():
        cls_preds = pd.read_csv(CLS_PREDS_PATH)
        cls_preds["date"] = pd.to_datetime(cls_preds["date"])

    print("\nGenerating plots …\n")

    print("[1/15] AQI distribution by county")
    plot_aqi_distribution(df)

    print("[2/15] County × month heatmap")
    plot_county_heatmap(df)

    print("[3/15] School hazardous event days by year")
    plot_hazardous_days_by_year(df)

    print("[4/15] AQI lag autocorrelation")
    plot_acf(df)

    print("[5/15] Feature importance (averaged across tree models)")
    plot_feature_importance()

    print("[6/15] Feature–target correlation")
    plot_feature_correlation(df)

    print("[7/15] Wildfire events overlaid on AQI spikes")
    plot_wildfire_aqi_overlay(df)

    print("[8/15] AQI vs meteorological variables")
    plot_met_scatter(df)

    if not metrics.empty:
        print("[9/15] Regression performance comparison")
        plot_regression_comparison(metrics)

        print("[10/15] Classification performance comparison")
        plot_classification_comparison(metrics)
    else:
        print("[9/15]  Skipping — no metrics.csv")
        print("[10/15] Skipping — no metrics.csv")

    if not cls_preds.empty:
        print("[11/15] ROC curves")
        plot_roc_curves(cls_preds)

        print("[12/15] Precision-recall curves")
        plot_precision_recall_curves(cls_preds)
    else:
        print("[11/15] Skipping — no classification_predictions.csv")
        print("[12/15] Skipping — no classification_predictions.csv")

    if not reg_preds.empty:
        print("[13/15] Predicted vs actual")
        plot_predicted_vs_actual(reg_preds)

        print("[14/15] Error by county")
        plot_error_by_county(reg_preds)

        print("[15/15] Error by season")
        plot_error_by_season(reg_preds)
    else:
        print("[13/15] Skipping — no regression_predictions.csv")
        print("[14/15] Skipping — no regression_predictions.csv")
        print("[15/15] Skipping — no regression_predictions.csv")

    print(f"\nAll plots saved to: results/plots/")


if __name__ == "__main__":
    main()
