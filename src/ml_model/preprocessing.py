"""
Preprocessing pipeline for Aether-Pulse ML model.

Accepts a pandas DataFrame as input (no dataset loading here).
Encodes categoricals, normalises numerics, and returns a feature matrix
ready for XGBoost.

Input features expected in the DataFrame:
    - age           : int/float — patient age in years
    - weight_kg     : float — body weight in kilograms
    - sex           : str — "M" | "F" | "O" (other/unknown)
    - diagnosis_codes: str — space- or comma-separated ICD-10 codes (e.g. "E11 I10")
    - allergies     : str — comma-separated allergy names
    - current_medications: str — comma-separated drug names
    - creatinine    : float — serum creatinine in mg/dL (renal marker)
    - alt           : float — alanine aminotransferase in U/L (hepatic marker)
    - label         : int (optional, 0=safe, 1=dangerous) — for training only
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEX_MAP: dict[str, int] = {"m": 0, "male": 0, "f": 1, "female": 1, "o": 2, "other": 2, "unknown": 2}

# Known allergy tokens used as binary features
KNOWN_ALLERGIES: list[str] = [
    "penicillin", "sulfa", "aspirin", "codeine", "contrast",
    "latex", "iodine", "nsaid", "peanut",
]

# Known drug tokens used as binary features
KNOWN_DRUGS: list[str] = [
    "warfarin", "aspirin", "metformin", "digoxin", "lithium",
    "ssri", "maoi", "ibuprofen", "amiodarone", "simvastatin",
    "methotrexate", "nsaid", "clopidogrel", "omeprazole",
    "acetaminophen", "paracetamol",
]

# Numeric feature columns (will be normalised)
NUMERIC_COLS: list[str] = ["age", "weight_kg", "creatinine", "alt"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: Any) -> list[str]:
    """Split a free-text field into lowercase tokens."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [t.strip().lower() for t in re.split(r"[,\s]+", text) if t.strip()]


def _encode_sex(value: Any) -> int:
    """Map sex string to integer code (0=M, 1=F, 2=other/unknown)."""
    if pd.isna(value):
        return 2
    return SEX_MAP.get(str(value).strip().lower(), 2)


def _allergy_features(series: pd.Series) -> pd.DataFrame:
    """Create binary columns for known allergens."""
    result = {}
    for allergen in KNOWN_ALLERGIES:
        result[f"allergy_{allergen}"] = series.apply(
            lambda x: int(allergen in _tokenize(x))
        )
    # Total allergy count as an additional numeric signal
    result["allergy_count"] = series.apply(lambda x: len(_tokenize(x)))
    return pd.DataFrame(result)


def _drug_features(series: pd.Series) -> pd.DataFrame:
    """Create binary columns for known current medications."""
    result = {}
    for drug in KNOWN_DRUGS:
        result[f"drug_{drug}"] = series.apply(
            lambda x: int(drug in _tokenize(x))
        )
    result["drug_count"] = series.apply(lambda x: len(_tokenize(x)))
    return pd.DataFrame(result)


def _diagnosis_features(series: pd.Series) -> pd.DataFrame:
    """
    Create a simple numeric feature from diagnosis codes:
    - Number of distinct codes
    - Binary flag for common high-risk prefixes (E=endocrine, I=circulatory, N=renal)
    """
    result: dict[str, list[int | float]] = {
        "diag_count": [],
        "diag_endocrine": [],
        "diag_circulatory": [],
        "diag_renal": [],
    }
    for text in series:
        codes = _tokenize(text) if not pd.isna(text) else []
        result["diag_count"].append(len(codes))
        result["diag_endocrine"].append(int(any(c.startswith("e") for c in codes)))
        result["diag_circulatory"].append(int(any(c.startswith("i") for c in codes)))
        result["diag_renal"].append(int(any(c.startswith("n") for c in codes)))
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MedicationPreprocessor:
    """
    Fit-transform pipeline for patient safety feature engineering.

    Usage::

        preprocessor = MedicationPreprocessor()
        X = preprocessor.fit_transform(df_train)
        X_test = preprocessor.transform(df_test)
    """

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessor on training data and transform it.

        Args:
            df: pandas DataFrame with expected columns (see module docstring).

        Returns:
            2-D numpy array of shape (n_samples, n_features).
        """
        feature_df = self._build_features(df)
        numeric_block = feature_df[NUMERIC_COLS].to_numpy(dtype=np.float32)
        numeric_scaled = self._scaler.fit_transform(numeric_block)
        self._fitted = True
        return self._concat(feature_df, numeric_scaled)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the already-fitted preprocessor.

        Args:
            df: pandas DataFrame with expected columns.

        Returns:
            2-D numpy array of shape (n_samples, n_features).

        Raises:
            RuntimeError: if called before ``fit_transform``.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform(). Use fit_transform() first.")
        feature_df = self._build_features(df)
        numeric_block = feature_df[NUMERIC_COLS].to_numpy(dtype=np.float32)
        numeric_scaled = self._scaler.transform(numeric_block)
        return self._concat(feature_df, numeric_scaled)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assemble the raw (unscaled) feature DataFrame."""
        df = df.copy()

        # Fill missing numerics with sensible defaults
        df["age"] = pd.to_numeric(df.get("age"), errors="coerce").fillna(0)
        df["weight_kg"] = pd.to_numeric(df.get("weight_kg"), errors="coerce").fillna(0)
        df["creatinine"] = pd.to_numeric(df.get("creatinine"), errors="coerce").fillna(1.0)
        df["alt"] = pd.to_numeric(df.get("alt"), errors="coerce").fillna(20.0)

        # Categorical: sex
        df["sex_enc"] = df.get("sex", pd.Series(["unknown"] * len(df))).apply(_encode_sex)

        # Structured text features
        allergy_df = _allergy_features(df.get("allergies", pd.Series([""] * len(df))))
        drug_df = _drug_features(df.get("current_medications", pd.Series([""] * len(df))))
        diag_df = _diagnosis_features(df.get("diagnosis_codes", pd.Series([""] * len(df))))

        feature_df = pd.concat(
            [df[NUMERIC_COLS], df[["sex_enc"]], allergy_df, drug_df, diag_df],
            axis=1,
        )
        return feature_df

    @staticmethod
    def _concat(feature_df: pd.DataFrame, numeric_scaled: np.ndarray) -> np.ndarray:
        """Replace the raw numeric columns with their scaled counterparts and stack."""
        non_numeric = feature_df.drop(columns=NUMERIC_COLS).to_numpy(dtype=np.float32)
        return np.hstack([numeric_scaled, non_numeric])

    def get_feature_names(self) -> list[str]:
        """Return feature names in the order they appear in the output array."""
        base = NUMERIC_COLS.copy()
        base += ["sex_enc"]
        base += [f"allergy_{a}" for a in KNOWN_ALLERGIES] + ["allergy_count"]
        base += [f"drug_{d}" for d in KNOWN_DRUGS] + ["drug_count"]
        base += ["diag_count", "diag_endocrine", "diag_circulatory", "diag_renal"]
        return base
