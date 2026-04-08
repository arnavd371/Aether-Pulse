"""
Hybrid decision engine for Aether-Pulse CDSS.

Logic:
1. Run deterministic rule engine.
   - If any rule is flagged → return immediately with source="rules".
2. Run XGBoost ML model.
   - If ML confidence ≥ threshold → return ML result with source="ml".
3. Fallback: run gradient boosting only (reuse same XGBoost model with
   adjusted probability interpretation) with source="fallback".

Output schema::

    {
        "safe": bool,
        "flags": [{"flagged": bool, "reason": str, "severity": str}, ...],
        "confidence": float,
        "source": "rules" | "ml" | "fallback",
    }
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Allow importing sibling packages when run directly
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules_engine.safety_rules import run_all_rules, any_flagged  # noqa: E402
from ml_model.preprocessing import MedicationPreprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Hybrid rule + ML safety decision engine.

    Args:
        model_path: Path to the joblib artefact produced by ``train.py``.
                    The artefact must be a dict with keys ``"model"`` and
                    ``"preprocessor"``.
        ml_confidence_threshold: Minimum ML probability to trust the ML output.
                                  Below this threshold the engine falls back to
                                  the gradient boosting model directly (same
                                  underlying model, different confidence band).
    """

    def __init__(
        self,
        model_path: str | None = None,
        ml_confidence_threshold: float = 0.6,
    ) -> None:
        self._ml_threshold = ml_confidence_threshold
        self._model: Any = None
        self._preprocessor: MedicationPreprocessor | None = None

        if model_path and os.path.isfile(model_path):
            self._load_model(model_path)

    # ------------------------------------------------------------------
    @property
    def is_model_loaded(self) -> bool:
        """Return True if the ML model and preprocessor are loaded and ready."""
        return self._model is not None and self._preprocessor is not None

    # ------------------------------------------------------------------
    def _load_model(self, model_path: str) -> None:
        artefact = joblib.load(model_path)
        self._model = artefact["model"]
        self._preprocessor = artefact["preprocessor"]

    # ------------------------------------------------------------------
  def _patient_to_dataframe(
        self,
        patient: dict[str, Any],
        medication: str,
        dose_mg: float | None,
    ) -> pd.DataFrame:
        row = {
            "age": patient.get("age", 0),
            "weight_kg": patient.get("weight_kg", 0),
            "sex": patient.get("sex", "unknown"),
            "diagnosis_codes": " ".join(patient.get("diagnosis_codes", [])),
            "allergies": ", ".join(patient.get("allergies", [])),
            "current_medications": ", ".join(patient.get("current_medications", [])),
            "creatinine": patient.get("creatinine", 1.0),
            "alt": patient.get("alt", 20.0),
            # ADD THESE:
            "Recommended_Medication": medication,
            "Dosage": dose_mg if dose_mg else 0
        }
        return pd.DataFrame([row])

    # ------------------------------------------------------------------
    def decide(
        self,
        patient: dict[str, Any],
        medication: str,
        dose_mg: float | None = None,
    ) -> dict[str, Any]:
        """
        Run the full hybrid decision pipeline.

        Args:
            patient: Patient profile dict (see API schema for keys).
            medication: Name of the medication to evaluate.
            dose_mg: Prescribed dose in milligrams (optional).

        Returns:
            Decision dict with keys: ``safe``, ``flags``, ``confidence``, ``source``.
        """
        # ------------------------------------------------------------------
        # Step 1: deterministic rule engine
        # ------------------------------------------------------------------
        rule_results = run_all_rules(patient, medication, dose_mg)
        flagged_rules = [r for r in rule_results if r["flagged"]]

        if flagged_rules:
            return {
                "safe": False,
                "flags": flagged_rules,
                "confidence": 1.0,
                "source": "rules",
            }

        # ------------------------------------------------------------------
        # Step 2: ML model
        # ------------------------------------------------------------------
        if self._model is None or self._preprocessor is None:
            # No model loaded — default to safe with low confidence
            return {
                "safe": True,
                "flags": [],
                "confidence": 0.0,
                "source": "fallback",
            }

        df = self._patient_to_dataframe(patient, medication, dose_mg)
        X = self._preprocessor.transform(df)
        proba_dangerous = float(self._model.predict_proba(X)[0, 1])

        if proba_dangerous >= self._ml_threshold:
            return {
                "safe": False,
                "flags": [
                    {
                        "flagged": True,
                        "reason": f"ML model flagged as dangerous (confidence {proba_dangerous:.2f})",
                        "severity": "high" if proba_dangerous >= 0.8 else "medium",
                    }
                ],
                "confidence": proba_dangerous,
                "source": "ml",
            }

        if (1.0 - proba_dangerous) >= self._ml_threshold:
            return {
                "safe": True,
                "flags": [],
                "confidence": 1.0 - proba_dangerous,
                "source": "ml",
            }

        # ------------------------------------------------------------------
        # Step 3: fallback — gradient boosting only (same model, low-confidence band)
        # ------------------------------------------------------------------
        # Re-scale the distance from the decision boundary (0.5) to a [0, 1]
        # confidence score: a probability of 0.5 → confidence 0.0, while
        # probabilities at 0.0 or 1.0 → confidence 1.0.
        return {
            "safe": proba_dangerous < 0.5,
            "flags": [],
            "confidence": abs(proba_dangerous - 0.5) * 2,
            "source": "fallback",
        }
