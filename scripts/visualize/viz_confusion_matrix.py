"""Confusion matrix for the exceedance classification model (test set).

The critical failure mode is false negatives — days the model predicted
as safe but were actually unhealthy. Those are the missed warnings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDS_PATH = PROJECT_ROOT / "results" / "models" / "classification_predictions.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "plots" / "confusion_matrix.png"


def main() -> None:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions: {PREDS_PATH}")

    df = pd.read_csv(PREDS_PATH)
    required = {"target_next_day_exceedance", "prediction_next_day_exceedance"}
    if not required.issubset(df.columns):
        print(f"Predictions file missing columns: {required - set(df.columns)}")
        return

    df = df.dropna(subset=list(required))
    if df.empty:
        print("No prediction rows available.")
        return

    y_true = df["target_next_day_exceedance"].astype(int)
    y_pred = df["prediction_next_day_exceedance"].astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = cm.sum()

    labels = [
        [f"True Negative\n{cm[0,0]}\n({cm[0,0]/total:.1%})",
         f"False Positive\n{cm[0,1]}\n({cm[0,1]/total:.1%})\n← False alarm"],
        [f"False Negative\n{cm[1,0]}\n({cm[1,0]/total:.1%})\n← Missed warning",
         f"True Positive\n{cm[1,1]}\n({cm[1,1]/total:.1%})"],
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        ax=ax,
        annot=[[labels[i][j] for j in range(2)] for i in range(2)],
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1,
        xticklabels=["Predicted: Safe (0)", "Predicted: Unhealthy (1)"],
        yticklabels=["Actual: Safe (0)", "Actual: Unhealthy (1)"],
        annot_kws={"size": 10},
    )
    ax.set_title("Confusion Matrix — Exceedance Classification (Test Set)\n"
                 f"Threshold: AQI ≥ 120  |  n = {total}", fontweight="bold")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
