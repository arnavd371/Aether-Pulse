"""
XGBoost training script for Aether-Pulse CDSS.

Designed to run in Google Colab or locally.
The path to the dataset CSV/Parquet is supplied as a CLI argument or
via the AETHER_DATA_PATH environment variable.

Usage (CLI)::

    python train.py --data_path /path/to/dataset.csv --output_path model.pkl

Usage (Colab)::

    !python src/ml_model/train.py --data_path /content/drive/MyDrive/aether_data.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model.preprocessing import MedicationPreprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# Default hyper-parameter grid for GridSearchCV
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}


# ---------------------------------------------------------------------------
# Optuna objective (optional — used when --tuner optuna is passed)
# ---------------------------------------------------------------------------

def _optuna_objective(trial, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold) -> float:
    """Optuna objective: maximise mean cross-validated AUC-ROC."""
    import optuna  # noqa: PLC0415 — optional import

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    model = XGBClassifier(**params)
    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], proba))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load CSV or Parquet dataset from the given path."""
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def train(
    data_path: str,
    output_path: str = "model.pkl",
    tuner: str = "grid",
    n_trials: int = 30,
    label_col: str = "label",
) -> None:
    """
    Full training pipeline.

    Args:
        data_path: Path to the training dataset (CSV or Parquet).
        output_path: Where to save the trained model (joblib pickle).
        tuner: ``"grid"`` for GridSearchCV, ``"optuna"`` for Optuna.
        n_trials: Number of Optuna trials (only used when tuner=="optuna").
        label_col: Name of the binary target column in the dataset.
    """
    print(f"[Aether-Pulse] Loading dataset from: {data_path}")
    df = load_dataset(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset. Columns: {list(df.columns)}")

    y = df[label_col].astype(int).to_numpy()
    X_raw = df.drop(columns=[label_col])

    print(f"[Aether-Pulse] Dataset shape: {df.shape}  |  Class balance: {np.bincount(y)}")

    # Preprocessing
    preprocessor = MedicationPreprocessor()
    X = preprocessor.fit_transform(X_raw)
    print(f"[Aether-Pulse] Feature matrix shape: {X.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Hyper-parameter tuning
    if tuner == "optuna":
        try:
            import optuna  # noqa: PLC0415
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: _optuna_objective(trial, X, y, cv),
                n_trials=n_trials,
                show_progress_bar=False,
            )
            best_params = study.best_params
            best_params.update({"use_label_encoder": False, "eval_metric": "logloss", "random_state": 42})
            print(f"[Aether-Pulse] Best Optuna params: {best_params}")
            best_model = XGBClassifier(**best_params)
        except ImportError:
            print("[Aether-Pulse] Optuna not installed — falling back to GridSearchCV.")
            tuner = "grid"

    if tuner == "grid":
        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        search = GridSearchCV(
            base_model,
            PARAM_GRID,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        print(f"[Aether-Pulse] Best GridSearch params: {search.best_params_}")

    # Final cross-validated evaluation
    print("\n[Aether-Pulse] Cross-validated metrics (5-fold):")
    acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        best_model.fit(X[train_idx], y[train_idx])
        preds = best_model.predict(X[val_idx])
        proba = best_model.predict_proba(X[val_idx])[:, 1]
        acc_list.append(accuracy_score(y[val_idx], preds))
        prec_list.append(precision_score(y[val_idx], preds, zero_division=0))
        rec_list.append(recall_score(y[val_idx], preds, zero_division=0))
        f1_list.append(f1_score(y[val_idx], preds, zero_division=0))
        auc_list.append(roc_auc_score(y[val_idx], proba))
        print(
            f"  Fold {fold}: Acc={acc_list[-1]:.4f}  Prec={prec_list[-1]:.4f}"
            f"  Rec={rec_list[-1]:.4f}  F1={f1_list[-1]:.4f}  AUC={auc_list[-1]:.4f}"
        )

    print(
        f"\n  Mean — Acc={np.mean(acc_list):.4f}  Prec={np.mean(prec_list):.4f}"
        f"  Rec={np.mean(rec_list):.4f}  F1={np.mean(f1_list):.4f}  AUC={np.mean(auc_list):.4f}"
    )

    # Re-fit on full dataset
    best_model.fit(X, y)

    # Save model + preprocessor
    artefact = {"model": best_model, "preprocessor": preprocessor}
    joblib.dump(artefact, output_path)
    print(f"\n[Aether-Pulse] Model saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aether-Pulse XGBoost training script")
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.environ.get("AETHER_DATA_PATH", ""),
        help="Path to the training dataset (CSV or Parquet). "
             "Can also be set via the AETHER_DATA_PATH environment variable.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="model.pkl",
        help="Output path for the saved model artefact (default: model.pkl).",
    )
    parser.add_argument(
        "--tuner",
        type=str,
        choices=["grid", "optuna"],
        default="grid",
        help="Hyper-parameter tuner: 'grid' (GridSearchCV) or 'optuna' (default: grid).",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=30,
        help="Number of Optuna trials (only used when --tuner optuna, default: 30).",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the target column in the dataset (default: label).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if not args.data_path:
        print("ERROR: --data_path is required (or set AETHER_DATA_PATH env var).")
        sys.exit(1)
    train(
        data_path=args.data_path,
        output_path=args.output_path,
        tuner=args.tuner,
        n_trials=args.n_trials,
        label_col=args.label_col,
    )
