"""Integration tests for the Aether-Pulse FastAPI application."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "ok"


class TestCheckMedicationEndpoint:
    def _safe_payload(self, medication: str = "metformin", dose_mg: float | None = None) -> dict:
        payload: dict = {
            "patient": {
                "age": 40,
                "weight_kg": 75,
                "sex": "M",
                "diagnosis_codes": ["E11"],
                "allergies": [],
                "current_medications": ["lisinopril"],
                "creatinine": 0.9,
                "alt": 22,
            },
            "medication": medication,
        }
        if dose_mg is not None:
            payload["dose_mg"] = dose_mg
        return payload

    def test_safe_patient_returns_200(self):
        response = client.post("/check-medication", json=self._safe_payload())
        assert response.status_code == 200

    def test_response_schema(self):
        response = client.post("/check-medication", json=self._safe_payload())
        data = response.json()
        assert "safe" in data
        assert "flags" in data
        assert "confidence" in data
        assert "source" in data

    def test_allergy_patient_flagged(self):
        payload = self._safe_payload("aspirin")
        payload["patient"]["allergies"] = ["aspirin"]
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safe"] is False
        assert data["source"] == "rules"

    def test_child_aspirin_flagged(self):
        payload = self._safe_payload("aspirin")
        payload["patient"]["age"] = 7
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safe"] is False
        assert data["source"] == "rules"

    def test_drug_interaction_flagged(self):
        payload = self._safe_payload("aspirin")
        payload["patient"]["current_medications"] = ["warfarin"]
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safe"] is False

    def test_renal_impairment_metformin_flagged(self):
        payload = self._safe_payload("metformin")
        payload["patient"]["creatinine"] = 3.0
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safe"] is False

    def test_invalid_sex_returns_422(self):
        payload = self._safe_payload()
        payload["patient"]["sex"] = "alien"
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 422

    def test_negative_age_returns_422(self):
        payload = self._safe_payload()
        payload["patient"]["age"] = -5
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 422

    def test_confidence_in_range(self):
        response = client.post("/check-medication", json=self._safe_payload())
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_pediatric_overdose_flagged(self):
        payload = self._safe_payload("paracetamol", dose_mg=1000)
        payload["patient"]["age"] = 5
        payload["patient"]["weight_kg"] = 20  # 50 mg/kg >> 15 mg/kg threshold
        response = client.post("/check-medication", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["safe"] is False
