"""Train and evaluate next-day AQI regression and exceedance classification models.

Models trained:
  - Persistence baseline  (predict tomorrow = today's AQI / exceedance)
  - Ridge regression      + Logistic regression   (linear, scaled)
  - Random Forest         (tree ensemble)
  - XGBoost               (gradient boosting, optional — requires libomp)
  - LightGBM              (gradient boosting, optional — requires libomp)

XGBoost and LightGBM are skipped gracefully if their native libraries are not
installed (Mac: run `brew install libomp` to enable them).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional gradient-boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except Exception:
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _LGBM_AVAILABLE = True
except Exception:
    _LGBM_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.date_config import get_train_val_test_ranges


FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "modeling" / "features_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "results" / "models"

EXCEEDANCE_THRESHOLD = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_split(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Filter rows within an inclusive date window."""
    return frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()


def _metrics_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _metrics_classification(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def _scaled(model):
    """Wrap a model in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Train all models on the train split and evaluate on val/test time splits."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature dataset: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    required = {"county", "date", "target_next_day_aqi", "target_next_day_exceedance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature dataset missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    excluded = {"county", "date", "target_next_day_aqi", "target_next_day_exceedance"}
    feature_cols = [
        col for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].notna().any()
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available for model training.")

    model_frame = df.dropna(
        subset=feature_cols + ["target_next_day_aqi", "target_next_day_exceedance", "date"]
    ).copy()
    model_frame["target_next_day_exceedance"] = model_frame["target_next_day_exceedance"].astype(int)

    ranges = get_train_val_test_ranges()
    train_start, train_end = [pd.Timestamp(d) for d in ranges["train"]]
    val_start, val_end = [pd.Timestamp(d) for d in ranges["val"]]
    test_start, test_end = [pd.Timestamp(d) for d in ranges["test"]]

    train_df = _filter_split(model_frame, train_start, train_end)
    val_df = _filter_split(model_frame, val_start, val_end)
    test_df = _filter_split(model_frame, test_start, test_end)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One or more time splits are empty. Check dates_config.json and available data range.")

    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    x_test = test_df[feature_cols]

    y_train_reg = train_df["target_next_day_aqi"]
    y_val_reg = val_df["target_next_day_aqi"]
    y_test_reg = test_df["target_next_day_aqi"]

    y_train_cls = train_df["target_next_day_exceedance"]
    y_val_cls = val_df["target_next_day_exceedance"]
    y_test_cls = test_df["target_next_day_exceedance"]

    multiclass = len(np.unique(y_train_cls)) > 1

    # ------------------------------------------------------------------
    # Model registry: list of (name, regressor, classifier)
    # ------------------------------------------------------------------
    reg_models: list[tuple[str, object]] = [
        ("ridge", _scaled(Ridge(alpha=1.0))),
        ("random_forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ]
    cls_models: list[tuple[str, object]] = [
        ("logistic", _scaled(LogisticRegression(max_iter=1000, random_state=42))),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
    ]

    if _XGBOOST_AVAILABLE:
        reg_models.append(("xgboost", XGBRegressor(n_estimators=300, random_state=42,
                                                    verbosity=0, n_jobs=-1)))
        cls_models.append(("xgboost", XGBClassifier(n_estimators=300, random_state=42,
                                                     verbosity=0, use_label_encoder=False,
                                                     eval_metric="logloss", n_jobs=-1)))
    else:
        print("XGBoost skipped (libomp not installed — run `brew install libomp` to enable).")

    if _LGBM_AVAILABLE:
        reg_models.append(("lightgbm", LGBMRegressor(n_estimators=300, random_state=42,
                                                      n_jobs=-1, verbose=-1)))
        cls_models.append(("lightgbm", LGBMClassifier(n_estimators=300, random_state=42,
                                                       n_jobs=-1, verbose=-1)))
    else:
        print("LightGBM skipped (libomp not installed — run `brew install libomp` to enable).")

    # ------------------------------------------------------------------
    # Persistence baseline (no model object needed)
    # ------------------------------------------------------------------
    persistence_col = "aqi_lag_1"
    if persistence_col not in feature_cols:
        print(f"Warning: '{persistence_col}' not in features, skipping persistence baseline.")
        persistence_col = None

    # ------------------------------------------------------------------
    # Train and evaluate all models
    # ------------------------------------------------------------------
    all_metrics_rows: list[dict] = []
    fitted_reg: dict[str, object] = {}
    fitted_cls: dict[str, object] = {}

    # Regression
    for name, model in reg_models:
        model.fit(x_train, y_train_reg)
        fitted_reg[name] = model
        for split_name, x_s, y_s in [("validation", x_val, y_val_reg), ("test", x_test, y_test_reg)]:
            m = _metrics_regression(y_s, model.predict(x_s))
            all_metrics_rows.append({"split": split_name, "task": "regression", "model": name, **m})

    if persistence_col:
        for split_name, df_s, y_s in [("validation", val_df, y_val_reg), ("test", test_df, y_test_reg)]:
            m = _metrics_regression(y_s, df_s[persistence_col].values)
            all_metrics_rows.append({"split": split_name, "task": "regression", "model": "persistence", **m})

    # Classification
    for name, model in cls_models:
        if not multiclass:
            # Can't train a meaningful classifier when training set has only one class
            for split_name in ["validation", "test"]:
                all_metrics_rows.append({"split": split_name, "task": "classification", "model": name,
                                         "accuracy": float("nan"), "note": "single_class_in_train"})
            continue
        model.fit(x_train, y_train_cls)
        fitted_cls[name] = model
        for split_name, x_s, y_s in [("validation", x_val, y_val_cls), ("test", x_test, y_test_cls)]:
            proba = None
            if hasattr(model, "predict_proba") and len(np.unique(y_s)) > 1:
                proba = model.predict_proba(x_s)[:, 1]
            m = _metrics_classification(y_s, model.predict(x_s), proba)
            all_metrics_rows.append({"split": split_name, "task": "classification", "model": name, **m})

    if persistence_col:
        for split_name, df_s, y_s in [("validation", val_df, y_val_cls), ("test", test_df, y_test_cls)]:
            persist_cls = (df_s[persistence_col].values >= EXCEEDANCE_THRESHOLD).astype(int)
            m = _metrics_classification(y_s, persist_cls, None)
            all_metrics_rows.append({"split": split_name, "task": "classification", "model": "persistence", **m})

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Primary RF models kept at the original paths for viz script compatibility
    rf_reg = fitted_reg["random_forest"]
    joblib.dump(rf_reg, MODELS_DIR / "regression_model.joblib")

    if "random_forest" in fitted_cls:
        rf_cls = fitted_cls["random_forest"]
        joblib.dump(rf_cls, MODELS_DIR / "classification_model.joblib")

    # All other fitted models
    for name, model in fitted_reg.items():
        if name != "random_forest":
            joblib.dump(model, MODELS_DIR / f"{name}_regression_model.joblib")
    for name, model in fitted_cls.items():
        if name != "random_forest":
            joblib.dump(model, MODELS_DIR / f"{name}_classification_model.joblib")

    # Regression predictions (RF + all others, one column per model)
    reg_preds_df = test_df[["county", "date", "target_next_day_aqi"]].copy()
    reg_preds_df["date"] = reg_preds_df["date"].dt.strftime("%Y-%m-%d")
    for name, model in fitted_reg.items():
        col = "prediction_next_day_aqi" if name == "random_forest" else f"pred_aqi_{name}"
        reg_preds_df[col] = model.predict(x_test)
    if persistence_col:
        reg_preds_df["pred_aqi_persistence"] = test_df[persistence_col].values
    reg_preds_df.to_csv(MODELS_DIR / "regression_predictions.csv", index=False)

    # Classification predictions (RF + all others)
    cls_preds_df = test_df[["county", "date", "target_next_day_exceedance"]].copy()
    cls_preds_df["date"] = cls_preds_df["date"].dt.strftime("%Y-%m-%d")
    for name, model in fitted_cls.items():
        col = "prediction_next_day_exceedance" if name == "random_forest" else f"pred_exc_{name}"
        cls_preds_df[col] = model.predict(x_test)
        if hasattr(model, "predict_proba"):
            prob_col = "prediction_probability_exceedance" if name == "random_forest" else f"prob_exc_{name}"
            cls_preds_df[prob_col] = model.predict_proba(x_test)[:, 1]
    if persistence_col:
        cls_preds_df["pred_exc_persistence"] = (test_df[persistence_col].values >= EXCEEDANCE_THRESHOLD).astype(int)
    cls_preds_df.to_csv(MODELS_DIR / "classification_predictions.csv", index=False)

    # Metrics
    metrics_table = pd.DataFrame(all_metrics_rows)
    metrics_table.to_csv(MODELS_DIR / "metrics.csv", index=False)

    # Build RF-only metrics for backward-compatible top-level keys
    def _pick(task: str, model: str, split: str) -> dict:
        for row in all_metrics_rows:
            if row.get("task") == task and row.get("model") == model and row.get("split") == split:
                return {k: v for k, v in row.items() if k not in {"task", "model", "split"}}
        return {}

    metrics_payload = {
        "feature_columns": feature_cols,
        "split_rows": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        # Backward-compatible: RF model metrics at top level (used by existing tests/integrations)
        "validation": {
            "regression": _pick("regression", "random_forest", "validation"),
            "classification": _pick("classification", "random_forest", "validation"),
        },
        "test": {
            "regression": _pick("regression", "random_forest", "test"),
            "classification": _pick("classification", "random_forest", "test"),
        },
        # Full comparison across all models
        "all_models": all_metrics_rows,
    }
    with (MODELS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    print(f"\nWrote model artifacts to: {MODELS_DIR}")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
