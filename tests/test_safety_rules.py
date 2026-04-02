"""Unit tests for the Aether-Pulse safety rules engine."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from rules_engine.safety_rules import (
    check_allergy_contraindication,
    check_age_contraindication,
    check_drug_drug_interactions,
    check_pediatric_weight_dose,
    check_renal_hepatic_impairment,
    run_all_rules,
    any_flagged,
)


# ---------------------------------------------------------------------------
# Allergy contraindication
# ---------------------------------------------------------------------------

class TestAllergyContraindication:
    def test_direct_allergy_flagged(self):
        patient = {"allergies": ["aspirin"]}
        result = check_allergy_contraindication(patient, "aspirin")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_cross_reactivity_flagged(self):
        patient = {"allergies": ["penicillin"]}
        result = check_allergy_contraindication(patient, "amoxicillin")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_no_allergy_safe(self):
        patient = {"allergies": ["penicillin"]}
        result = check_allergy_contraindication(patient, "metformin")
        assert result["flagged"] is False

    def test_empty_allergies_safe(self):
        patient = {"allergies": []}
        result = check_allergy_contraindication(patient, "ibuprofen")
        assert result["flagged"] is False

    def test_case_insensitive(self):
        patient = {"allergies": ["Sulfa"]}
        result = check_allergy_contraindication(patient, "Bactrim")
        assert result["flagged"] is True


# ---------------------------------------------------------------------------
# Pediatric weight-based dosage
# ---------------------------------------------------------------------------

class TestPediatricWeightDose:
    def test_dose_exceeds_threshold_flagged(self):
        patient = {"age": 5, "weight_kg": 20}
        result = check_pediatric_weight_dose(patient, "paracetamol", 500)  # 25 mg/kg > 15
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_dose_within_threshold_safe(self):
        patient = {"age": 5, "weight_kg": 20}
        result = check_pediatric_weight_dose(patient, "paracetamol", 200)  # 10 mg/kg ≤ 15
        assert result["flagged"] is False

    def test_adult_patient_skipped(self):
        patient = {"age": 25, "weight_kg": 70}
        result = check_pediatric_weight_dose(patient, "paracetamol", 2000)
        assert result["flagged"] is False

    def test_no_weight_skipped(self):
        patient = {"age": 5, "weight_kg": None}
        result = check_pediatric_weight_dose(patient, "paracetamol", 500)
        assert result["flagged"] is False

    def test_unknown_drug_skipped(self):
        patient = {"age": 5, "weight_kg": 20}
        result = check_pediatric_weight_dose(patient, "unknowndrug", 999)
        assert result["flagged"] is False


# ---------------------------------------------------------------------------
# Renal / hepatic impairment
# ---------------------------------------------------------------------------

class TestRenalHepaticImpairment:
    def test_renal_impairment_metformin_flagged(self):
        patient = {"creatinine": 2.0}
        result = check_renal_hepatic_impairment(patient, "metformin")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_normal_creatinine_safe(self):
        patient = {"creatinine": 1.0}
        result = check_renal_hepatic_impairment(patient, "metformin")
        assert result["flagged"] is False

    def test_hepatic_impairment_paracetamol_flagged(self):
        patient = {"alt": 120}
        result = check_renal_hepatic_impairment(patient, "paracetamol")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_normal_alt_safe(self):
        patient = {"alt": 30}
        result = check_renal_hepatic_impairment(patient, "paracetamol")
        assert result["flagged"] is False

    def test_no_markers_safe(self):
        patient = {}
        result = check_renal_hepatic_impairment(patient, "metformin")
        assert result["flagged"] is False


# ---------------------------------------------------------------------------
# Drug-drug interactions
# ---------------------------------------------------------------------------

class TestDrugDrugInteractions:
    def test_warfarin_aspirin_flagged(self):
        patient = {"current_medications": ["warfarin"]}
        result = check_drug_drug_interactions(patient, "aspirin")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_no_interaction_safe(self):
        patient = {"current_medications": ["metformin"]}
        result = check_drug_drug_interactions(patient, "paracetamol")
        assert result["flagged"] is False

    def test_empty_medications_safe(self):
        patient = {"current_medications": []}
        result = check_drug_drug_interactions(patient, "warfarin")
        assert result["flagged"] is False

    def test_multiple_meds_worst_severity_returned(self):
        patient = {"current_medications": ["warfarin", "digoxin"]}
        # warfarin+aspirin = high, digoxin+amiodarone = medium
        result = check_drug_drug_interactions(patient, "aspirin")
        assert result["flagged"] is True
        assert result["severity"] == "high"


# ---------------------------------------------------------------------------
# Age-based contraindication
# ---------------------------------------------------------------------------

class TestAgeContraindication:
    def test_aspirin_child_flagged(self):
        patient = {"age": 10}
        result = check_age_contraindication(patient, "aspirin")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_aspirin_adult_safe(self):
        patient = {"age": 30}
        result = check_age_contraindication(patient, "aspirin")
        assert result["flagged"] is False

    def test_no_age_skipped(self):
        patient = {}
        result = check_age_contraindication(patient, "aspirin")
        assert result["flagged"] is False

    def test_tetracycline_young_child_flagged(self):
        patient = {"age": 5}
        result = check_age_contraindication(patient, "tetracycline")
        assert result["flagged"] is True


# ---------------------------------------------------------------------------
# run_all_rules / any_flagged integration
# ---------------------------------------------------------------------------

class TestRunAllRules:
    def test_safe_patient_no_flags(self):
        patient = {
            "age": 30,
            "weight_kg": 70,
            "sex": "F",
            "allergies": [],
            "current_medications": ["lisinopril"],
            "creatinine": 0.9,
            "alt": 25,
        }
        results = run_all_rules(patient, "metformin", dose_mg=500)
        assert not any_flagged(results)

    def test_allergy_triggers_flag(self):
        patient = {"allergies": ["aspirin"], "current_medications": [], "age": 30}
        results = run_all_rules(patient, "aspirin")
        assert any_flagged(results)

    def test_pediatric_dose_included_when_provided(self):
        patient = {"age": 5, "weight_kg": 20, "allergies": [], "current_medications": []}
        results = run_all_rules(patient, "paracetamol", dose_mg=600)  # 30 mg/kg
        assert any_flagged(results)

    def test_pediatric_dose_not_in_results_when_omitted(self):
        patient = {"age": 5, "weight_kg": 20, "allergies": [], "current_medications": []}
        results = run_all_rules(patient, "paracetamol")
        # Without dose_mg the pediatric check is not included so should be 4 rules
        assert len(results) == 4
