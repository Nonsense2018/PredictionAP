"""ROC curve for the exceedance classification model (test set).

Shows the tradeoff between catching bad air days (recall / true positive rate)
vs. raising false alarms (false positive rate) at different probability thresholds.
AUC closer to 1.0 = better discrimination.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDS_PATH = PROJECT_ROOT / "results" / "models" / "classification_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "roc_curve.png"


def main() -> None:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDS_PATH}")

    df = pd.read_csv(PREDS_PATH)

    if "prediction_probability_exceedance" not in df.columns:
        print("No probability scores in predictions file. ROC curve requires predict_proba output.")
        return

    required = {"target_next_day_exceedance", "prediction_probability_exceedance"}
    df = df.dropna(subset=list(required))

    if df.empty:
        print("No prediction rows available.")
        return

    if len(df["target_next_day_exceedance"].unique()) < 2:
        print("Only one class present in test labels — ROC curve is undefined for this data window.")
        return

    fpr, tpr, thresholds = roc_curve(
        df["target_next_day_exceedance"],
        df["prediction_probability_exceedance"],
    )
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier (AUC = 0.5)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")

    # Mark the threshold closest to the top-left corner (optimal operating point)
    distances = (fpr ** 2 + (1 - tpr) ** 2) ** 0.5
    best_idx = distances.argmin()
    ax.scatter(fpr[best_idx], tpr[best_idx], color="#EF5350", zorder=5, s=80,
               label=f"Optimal threshold ≈ {thresholds[best_idx]:.2f}")

    ax.set_xlabel("False Positive Rate (false alarms)")
    ax.set_ylabel("True Positive Rate (bad days caught)")
    ax.set_title("ROC Curve — Exceedance Classification (Test Set)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
