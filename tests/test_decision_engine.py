"""Unit tests for the Aether-Pulse hybrid decision engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from hybrid_engine.decision_engine import DecisionEngine


def _safe_patient() -> dict:
    return {
        "age": 40,
        "weight_kg": 75,
        "sex": "M",
        "diagnosis_codes": ["E11"],
        "allergies": [],
        "current_medications": ["lisinopril"],
        "creatinine": 0.9,
        "alt": 22,
    }


class TestDecisionEngine:
    def test_rules_override_no_model(self):
        """Rule flags should be returned even without a loaded model."""
        engine = DecisionEngine(model_path=None)
        patient = _safe_patient()
        patient["allergies"] = ["aspirin"]
        result = engine.decide(patient, "aspirin")
        assert result["safe"] is False
        assert result["source"] == "rules"
        assert len(result["flags"]) > 0

    def test_safe_patient_no_model_fallback(self):
        """Without a model, a safe patient should return source=fallback with safe=True."""
        engine = DecisionEngine(model_path=None)
        result = engine.decide(_safe_patient(), "metformin")
        assert result["source"] == "fallback"
        assert result["safe"] is True

    def test_child_aspirin_flagged_via_age_rule(self):
        engine = DecisionEngine(model_path=None)
        patient = _safe_patient()
        patient["age"] = 7
        result = engine.decide(patient, "aspirin")
        assert result["safe"] is False
        assert result["source"] == "rules"

    def test_renal_impairment_flagged_via_rules(self):
        engine = DecisionEngine(model_path=None)
        patient = _safe_patient()
        patient["creatinine"] = 3.0
        result = engine.decide(patient, "metformin")
        assert result["safe"] is False
        assert result["source"] == "rules"

    def test_drug_interaction_flagged_via_rules(self):
        engine = DecisionEngine(model_path=None)
        patient = _safe_patient()
        patient["current_medications"] = ["warfarin"]
        result = engine.decide(patient, "aspirin")
        assert result["safe"] is False
        assert result["source"] == "rules"

    def test_output_schema_keys(self):
        engine = DecisionEngine(model_path=None)
        result = engine.decide(_safe_patient(), "metformin")
        assert "safe" in result
        assert "flags" in result
        assert "confidence" in result
        assert "source" in result

    def test_confidence_in_range(self):
        engine = DecisionEngine(model_path=None)
        result = engine.decide(_safe_patient(), "metformin")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_nonexistent_model_path_handled(self):
        """Providing a non-existent model path should not raise — just no model loaded."""
        engine = DecisionEngine(model_path="/tmp/does_not_exist.pkl")
        result = engine.decide(_safe_patient(), "metformin")
        assert result["source"] == "fallback"
