"""
Blind validation and evaluation suite for the Aether-Pulse ML model.

Optimises for recall to minimise false negatives (dangerous cases
predicted as safe).

Usage::

    python evaluate.py --data_path /path/to/test_data.csv --model_path model.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matplotlib.use("Agg")  # non-interactive backend (Colab / headless)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _find_recall_optimised_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_precision: float = 0.7) -> float:
    """
    Find the probability threshold that maximises recall while keeping
    precision above *min_precision*.

    Returns the best threshold (defaulting to 0.5 if no threshold achieves
    the minimum precision constraint).
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_recall = 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_threshold = t

    return float(best_threshold)


def evaluate(
    data_path: str,
    model_path: str = "model.pkl",
    label_col: str = "label",
    output_dir: str = ".",
    optimise_recall: bool = True,
) -> dict[str, float]:
    """
    Load a test dataset and trained model artefact, compute metrics, and
    save evaluation plots.

    Args:
        data_path: Path to the blind test CSV or Parquet.
        model_path: Path to the joblib model artefact saved by train.py.
        label_col: Name of the target column.
        output_dir: Directory where plots (confusion matrix, ROC curve) are saved.
        optimise_recall: If True, find a threshold that maximises recall.

    Returns:
        Dict of metric names → float values.
    """
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"[Aether-Pulse Eval] Loading test data from: {data_path}")
    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found. Columns: {list(df.columns)}")

    y_true = df[label_col].astype(int).to_numpy()
    X_raw = df.drop(columns=[label_col])

    # ------------------------------------------------------------------
    # Load model artefact
    # ------------------------------------------------------------------
    print(f"[Aether-Pulse Eval] Loading model from: {model_path}")
    artefact = joblib.load(model_path)
    model = artefact["model"]
    preprocessor = artefact["preprocessor"]

    X = preprocessor.transform(X_raw)
    y_proba = model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Threshold selection
    # ------------------------------------------------------------------
    if optimise_recall:
        threshold = _find_recall_optimised_threshold(y_true, y_proba)
        print(f"[Aether-Pulse Eval] Recall-optimised threshold: {threshold:.2f}")
    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_proba),
        "threshold": threshold,
    }

    print("\n[Aether-Pulse Eval] Evaluation Results:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}  ← minimise false negatives")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.2f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn} (dangerous→safe miss!)  TP={tp}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Dangerous"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Aether-Pulse")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Aether-Pulse Eval] Confusion matrix saved to: {cm_path}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['auc_roc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Aether-Pulse")
    ax.legend()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Aether-Pulse Eval] ROC curve saved to: {roc_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aether-Pulse model evaluation script")
    parser.add_argument("--data_path", type=str, required=True, help="Path to blind test CSV or Parquet.")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path to trained model artefact.")
    parser.add_argument("--label_col", type=str, default="label", help="Target column name (default: label).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory for evaluation plots (default: current directory).",
    )
    parser.add_argument(
        "--no_recall_opt",
        action="store_true",
        help="Disable recall-optimised threshold selection (use 0.5 instead).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate(
        data_path=args.data_path,
        model_path=args.model_path,
        label_col=args.label_col,
        output_dir=args.output_dir,
        optimise_recall=not args.no_recall_opt,
    )
