"""Feature importance bar chart for both models.

Shows which inputs the Random Forests actually rely on.
Expected: lag_1 and roll7 dominate, wildfire distance matters in summer.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REG_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "regression_model.joblib"
CLS_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "classification_model.joblib"
METRICS_PATH = PROJECT_ROOT / "results" / "models" / "metrics.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "feature_importance.png"


def plot_importances(ax: plt.Axes, importances: np.ndarray, feature_names: list[str], title: str) -> None:
    sorted_idx = np.argsort(importances)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#1565C0",
        alpha=0.85,
    )
    ax.set_xlabel("Mean decrease in impurity (importance)")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    import json

    for path in [REG_MODEL_PATH, CLS_MODEL_PATH, METRICS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    with METRICS_PATH.open(encoding="utf-8") as f:
        metrics = json.load(f)

    feature_names = metrics.get("feature_columns", [])
    if not feature_names:
        print("No feature column names found in metrics.json.")
        return

    reg_model = joblib.load(REG_MODEL_PATH)
    cls_model = joblib.load(CLS_MODEL_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(feature_names) * 0.45)))

    plot_importances(axes[0], reg_model.feature_importances_, feature_names,
                     "Regression Model — Feature Importance\n(next-day AQI prediction)")
    plot_importances(axes[1], cls_model.feature_importances_, feature_names,
                     "Classification Model — Feature Importance\n(exceedance prediction)")

    fig.suptitle("Random Forest Feature Importances", fontsize=13, fontweight="bold")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
