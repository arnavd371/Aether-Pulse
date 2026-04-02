"""Unit tests for the Aether-Pulse preprocessing pipeline."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest

from ml_model.preprocessing import MedicationPreprocessor, NUMERIC_COLS


def _make_sample_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 5, 70, 45, 12],
            "weight_kg": [70, 20, 85, 60, 35],
            "sex": ["M", "F", "M", "F", "unknown"],
            "diagnosis_codes": ["E11 I10", "J06", "N18", "", "R50"],
            "allergies": ["penicillin", "", "sulfa", "aspirin, codeine", ""],
            "current_medications": ["metformin", "", "warfarin, digoxin", "aspirin", ""],
            "creatinine": [1.0, 0.5, 2.5, 1.2, 0.6],
            "alt": [20, 15, 90, 25, 18],
        }
    )[:n]


class TestMedicationPreprocessor:
    def test_fit_transform_shape(self):
        df = _make_sample_df(5)
        preprocessor = MedicationPreprocessor()
        X = preprocessor.fit_transform(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 5
        assert X.shape[1] > 0

    def test_transform_same_shape(self):
        df = _make_sample_df(5)
        preprocessor = MedicationPreprocessor()
        X_train = preprocessor.fit_transform(df)
        X_test = preprocessor.transform(_make_sample_df(3))
        assert X_train.shape[1] == X_test.shape[1]

    def test_transform_before_fit_raises(self):
        df = _make_sample_df(3)
        preprocessor = MedicationPreprocessor()
        with pytest.raises(RuntimeError):
            preprocessor.transform(df)

    def test_numeric_features_normalized(self):
        df = _make_sample_df(5)
        preprocessor = MedicationPreprocessor()
        X = preprocessor.fit_transform(df)
        # Scaled numeric columns should be in [0, 1]
        n_numeric = len(NUMERIC_COLS)
        assert X[:, :n_numeric].min() >= 0.0
        assert X[:, :n_numeric].max() <= 1.0

    def test_missing_values_handled(self):
        df = pd.DataFrame(
            {
                "age": [None, 30],
                "weight_kg": [None, 70],
                "sex": [None, "F"],
                "diagnosis_codes": [None, "E11"],
                "allergies": [None, "penicillin"],
                "current_medications": [None, "metformin"],
                "creatinine": [None, 1.0],
                "alt": [None, 20],
            }
        )
        preprocessor = MedicationPreprocessor()
        X = preprocessor.fit_transform(df)
        assert not np.any(np.isnan(X))

    def test_feature_names_count_matches(self):
        df = _make_sample_df(5)
        preprocessor = MedicationPreprocessor()
        X = preprocessor.fit_transform(df)
        names = preprocessor.get_feature_names()
        assert len(names) == X.shape[1]

    def test_sex_encoding(self):
        df = pd.DataFrame(
            {
                "age": [30, 30, 30],
                "weight_kg": [70, 70, 70],
                "sex": ["M", "F", "unknown"],
                "diagnosis_codes": ["", "", ""],
                "allergies": ["", "", ""],
                "current_medications": ["", "", ""],
                "creatinine": [1.0, 1.0, 1.0],
                "alt": [20.0, 20.0, 20.0],
            }
        )
        preprocessor = MedicationPreprocessor()
        X = preprocessor.fit_transform(df)
        # sex_enc column is index 4 (after 4 NUMERIC_COLS)
        sex_col_idx = len(NUMERIC_COLS)
        sex_values = X[:, sex_col_idx]
        assert set(sex_values.tolist()).issubset({0.0, 1.0, 2.0})
