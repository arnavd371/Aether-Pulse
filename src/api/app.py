"""
Aether-Pulse FastAPI REST API.

Endpoints:
    POST /check-medication  — patient profile + medication → safety flag
    GET  /health            — health check

Run locally::

    uvicorn src.api.app:app --reload

Run in Colab::

    !uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
"""

from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Allow sibling imports when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_engine.decision_engine import DecisionEngine  # noqa: E402

# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aether-Pulse CDSS API",
    description=(
        "Clinical Decision Support System — hybrid rule engine + XGBoost "
        "medication safety checker."
    ),
    version="1.0.0",
)

# Load model at startup (path configurable via env var).
_MODEL_PATH = os.environ.get("AETHER_MODEL_PATH", "model.pkl")
_engine = DecisionEngine(
    model_path=_MODEL_PATH if os.path.isfile(_MODEL_PATH) else None,
    ml_confidence_threshold=float(os.environ.get("AETHER_ML_THRESHOLD", "0.6")),
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PatientProfile(BaseModel):
    """Patient profile fields accepted by the API."""

    age: float = Field(..., ge=0, le=130, description="Patient age in years.")
    weight_kg: float | None = Field(None, ge=0, description="Body weight in kilograms.")
    sex: str = Field("unknown", description="Patient sex: 'M', 'F', or 'unknown'.")
    diagnosis_codes: list[str] = Field(
        default_factory=list,
        description="List of ICD-10 diagnosis codes (e.g. ['E11', 'I10']).",
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="Known drug/substance allergies.",
    )
    current_medications: list[str] = Field(
        default_factory=list,
        description="Medications the patient is currently taking.",
    )
    creatinine: float | None = Field(
        None,
        ge=0,
        description="Serum creatinine in mg/dL (renal function marker).",
    )
    alt: float | None = Field(
        None,
        ge=0,
        description="Alanine aminotransferase in U/L (hepatic function marker).",
    )

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        allowed = {"m", "male", "f", "female", "o", "other", "unknown"}
        if v.strip().lower() not in allowed:
            raise ValueError(f"sex must be one of {allowed}")
        return v.strip().lower()


class MedicationCheckRequest(BaseModel):
    """Request body for POST /check-medication."""

    patient: PatientProfile
    medication: str = Field(..., min_length=1, description="Name of the medication to evaluate.")
    dose_mg: float | None = Field(
        None,
        ge=0,
        description="Prescribed dose in milligrams (required for pediatric weight-based check).",
    )


class FlagDetail(BaseModel):
    """A single safety flag from the rule engine or ML model."""

    flagged: bool
    reason: str
    severity: str  # "low" | "medium" | "high"


class MedicationCheckResponse(BaseModel):
    """Response body for POST /check-medication."""

    safe: bool = Field(..., description="True if the medication is considered safe for this patient.")
    flags: list[FlagDetail] = Field(default_factory=list, description="List of safety flags raised.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0, 1].")
    source: str = Field(..., description="Decision source: 'rules', 'ml', or 'fallback'.")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return service health and whether the ML model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=_engine.is_model_loaded,
        version=app.version,
    )


@app.post(
    "/check-medication",
    response_model=MedicationCheckResponse,
    summary="Check medication safety for a patient",
)
async def check_medication(request: MedicationCheckRequest) -> MedicationCheckResponse:
    """
    Evaluate whether a medication is safe to prescribe to a given patient.

    The decision is made by:
    1. Running deterministic safety rules (allergies, DDIs, age/renal/hepatic checks).
    2. If all rules pass, running the XGBoost ML model.
    3. If ML confidence is below threshold, falling back to the gradient boosting model.

    Returns a unified safety decision with confidence score and decision source.
    """
    patient_dict: dict[str, Any] = request.patient.model_dump()

    try:
        result = _engine.decide(
            patient=patient_dict,
            medication=request.medication,
            dose_mg=request.dose_mg,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Decision engine error: {exc}") from exc

    return MedicationCheckResponse(
        safe=result["safe"],
        flags=[FlagDetail(**f) for f in result.get("flags", [])],
        confidence=result["confidence"],
        source=result["source"],
    )
