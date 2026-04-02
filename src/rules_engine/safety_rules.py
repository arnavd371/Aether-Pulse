"""
Deterministic safety rules engine for Aether-Pulse CDSS.

Each rule accepts a patient profile dict and medication info, and returns:
    {"flagged": bool, "reason": str, "severity": "low" | "medium" | "high"}
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Known allergy cross-reactivity / contraindication map
# Keys are allergy names (lowercased); values are sets of contraindicated drugs
# ---------------------------------------------------------------------------
ALLERGY_CONTRAINDICATIONS: dict[str, set[str]] = {
    "penicillin": {"amoxicillin", "ampicillin", "piperacillin", "oxacillin"},
    "sulfa": {"sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim"},
    "aspirin": {"ibuprofen", "naproxen", "indomethacin", "ketorolac"},
    "codeine": {"codeine", "morphine", "hydrocodone", "oxycodone"},
    "contrast": {"iodine"},
}

# ---------------------------------------------------------------------------
# Drug-drug interaction table
# Keys are frozensets of two drug names (lowercased)
# ---------------------------------------------------------------------------
DDI_TABLE: dict[frozenset[str], dict[str, Any]] = {
    frozenset({"warfarin", "aspirin"}): {
        "reason": "Warfarin + Aspirin: increased bleeding risk",
        "severity": "high",
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "reason": "Warfarin + Ibuprofen: increased bleeding risk",
        "severity": "high",
    },
    frozenset({"metformin", "contrast"}): {
        "reason": "Metformin + Iodine contrast: risk of lactic acidosis",
        "severity": "high",
    },
    frozenset({"ssri", "maoi"}): {
        "reason": "SSRI + MAOI: risk of serotonin syndrome",
        "severity": "high",
    },
    frozenset({"simvastatin", "amiodarone"}): {
        "reason": "Simvastatin + Amiodarone: increased risk of myopathy/rhabdomyolysis",
        "severity": "high",
    },
    frozenset({"methotrexate", "nsaid"}): {
        "reason": "Methotrexate + NSAID: reduced methotrexate clearance, toxicity risk",
        "severity": "high",
    },
    frozenset({"digoxin", "amiodarone"}): {
        "reason": "Digoxin + Amiodarone: digoxin toxicity risk",
        "severity": "medium",
    },
    frozenset({"lithium", "ibuprofen"}): {
        "reason": "Lithium + Ibuprofen: elevated lithium levels",
        "severity": "medium",
    },
    frozenset({"clopidogrel", "omeprazole"}): {
        "reason": "Clopidogrel + Omeprazole: reduced antiplatelet effect",
        "severity": "medium",
    },
}

# ---------------------------------------------------------------------------
# Age-based contraindications
# Format: {drug: [(min_age_excl, max_age_incl, reason, severity), ...]}
# Use None for open-ended bounds.
# ---------------------------------------------------------------------------
AGE_CONTRAINDICATIONS: dict[str, list[tuple[Any, Any, str, str]]] = {
    "aspirin": [(None, 18, "Aspirin in children <18 years: risk of Reye's syndrome", "high")],
    "tetracycline": [(None, 8, "Tetracycline in children <8 years: risk of tooth discoloration", "medium")],
    "fluoroquinolone": [(None, 18, "Fluoroquinolones in children <18 years: risk of tendinopathy", "medium")],
    "metformin": [(None, 10, "Metformin not recommended in children <10 years", "medium")],
    "warfarin": [(None, 0, "Warfarin generally contraindicated in neonates without specialist review", "high")],
}

# ---------------------------------------------------------------------------
# Weight-based pediatric dosage thresholds (mg/kg)
# {drug: max_dose_per_kg}
# ---------------------------------------------------------------------------
PEDIATRIC_DOSE_THRESHOLDS: dict[str, float] = {
    "paracetamol": 15.0,   # mg/kg per dose
    "acetaminophen": 15.0,
    "ibuprofen": 10.0,
    "amoxicillin": 25.0,
    "amoxicillin-clavulanate": 25.0,
    "metronidazole": 7.5,
    "trimethoprim-sulfamethoxazole": 5.0,
}

# ---------------------------------------------------------------------------
# Renal / hepatic impairment adjustments
# creatinine_threshold triggers "renal impairment" flag (mg/dL)
# ---------------------------------------------------------------------------
RENAL_SENSITIVE_DRUGS: dict[str, dict[str, Any]] = {
    "metformin": {"threshold": 1.5, "reason": "Metformin contraindicated in renal impairment — risk of lactic acidosis", "severity": "high"},
    "digoxin": {"threshold": 1.5, "reason": "Digoxin requires dose reduction in renal impairment", "severity": "medium"},
    "gabapentin": {"threshold": 2.0, "reason": "Gabapentin requires dose reduction in renal impairment", "severity": "medium"},
    "lithium": {"threshold": 1.5, "reason": "Lithium toxic in renal impairment — close monitoring required", "severity": "high"},
    "nsaid": {"threshold": 1.5, "reason": "NSAIDs may worsen renal function in impaired patients", "severity": "medium"},
    "vancomycin": {"threshold": 1.5, "reason": "Vancomycin requires dose adjustment in renal impairment", "severity": "medium"},
}

HEPATIC_SENSITIVE_DRUGS: dict[str, dict[str, Any]] = {
    "acetaminophen": {"alt_threshold": 80, "reason": "Acetaminophen hepatotoxic in hepatic impairment — reduce dose", "severity": "high"},
    "paracetamol": {"alt_threshold": 80, "reason": "Paracetamol hepatotoxic in hepatic impairment — reduce dose", "severity": "high"},
    "statins": {"alt_threshold": 80, "reason": "Statins contraindicated in active hepatic disease", "severity": "high"},
    "methotrexate": {"alt_threshold": 40, "reason": "Methotrexate hepatotoxic — contraindicated in hepatic impairment", "severity": "high"},
    "isoniazid": {"alt_threshold": 40, "reason": "Isoniazid hepatotoxic — requires liver function monitoring", "severity": "medium"},
    "ketoconazole": {"alt_threshold": 40, "reason": "Ketoconazole hepatotoxic — avoid in hepatic disease", "severity": "high"},
}


def _normalize(name: str) -> str:
    """Lowercase and strip whitespace for consistent comparisons."""
    return name.strip().lower()


def check_allergy_contraindication(
    patient: dict[str, Any], medication: str
) -> dict[str, Any]:
    """
    Check whether the medication is contraindicated given patient allergies.

    Args:
        patient: dict with at least key ``allergies`` (list[str]).
        medication: name of the drug to check.

    Returns:
        Rule result dict.
    """
    med = _normalize(medication)
    allergies = [_normalize(a) for a in patient.get("allergies", [])]

    for allergy in allergies:
        # Direct match
        if allergy == med:
            return {
                "flagged": True,
                "reason": f"Patient is allergic to '{medication}'",
                "severity": "high",
            }
        # Cross-reactivity lookup
        contraindicated = ALLERGY_CONTRAINDICATIONS.get(allergy, set())
        if med in contraindicated:
            return {
                "flagged": True,
                "reason": (
                    f"Cross-reactivity: patient allergy to '{allergy}' "
                    f"contraindicates '{medication}'"
                ),
                "severity": "high",
            }

    return {"flagged": False, "reason": "No allergy contraindication detected", "severity": "low"}


def check_pediatric_weight_dose(
    patient: dict[str, Any], medication: str, dose_mg: float
) -> dict[str, Any]:
    """
    Validate weight-based pediatric dosage.

    Args:
        patient: dict with keys ``age`` (years) and ``weight_kg``.
        medication: drug name.
        dose_mg: prescribed dose in milligrams.

    Returns:
        Rule result dict.
    """
    age = patient.get("age", 18)
    weight_kg = patient.get("weight_kg")
    med = _normalize(medication)

    if age >= 18 or weight_kg is None or weight_kg <= 0:
        return {"flagged": False, "reason": "Pediatric dosage check not applicable", "severity": "low"}

    threshold = PEDIATRIC_DOSE_THRESHOLDS.get(med)
    if threshold is None:
        return {"flagged": False, "reason": f"No pediatric dosage threshold defined for '{medication}'", "severity": "low"}

    dose_per_kg = dose_mg / weight_kg
    if dose_per_kg > threshold:
        return {
            "flagged": True,
            "reason": (
                f"Pediatric dose exceeds threshold for '{medication}': "
                f"{dose_per_kg:.2f} mg/kg prescribed, max is {threshold} mg/kg"
            ),
            "severity": "high",
        }

    return {
        "flagged": False,
        "reason": f"Pediatric dosage within acceptable range ({dose_per_kg:.2f} mg/kg ≤ {threshold} mg/kg)",
        "severity": "low",
    }


def check_renal_hepatic_impairment(
    patient: dict[str, Any], medication: str
) -> dict[str, Any]:
    """
    Flag medications that require adjustment or are contraindicated in renal/hepatic impairment.

    Args:
        patient: dict with optional keys ``creatinine`` (mg/dL) and ``alt`` (U/L).
        medication: drug name.

    Returns:
        Rule result dict.
    """
    med = _normalize(medication)
    creatinine = patient.get("creatinine")
    alt = patient.get("alt")  # alanine aminotransferase as hepatic marker

    # Renal check
    if creatinine is not None:
        rule = RENAL_SENSITIVE_DRUGS.get(med)
        if rule and creatinine >= rule["threshold"]:
            return {
                "flagged": True,
                "reason": rule["reason"],
                "severity": rule["severity"],
            }

    # Hepatic check
    if alt is not None:
        rule = HEPATIC_SENSITIVE_DRUGS.get(med)
        if rule and alt >= rule["alt_threshold"]:
            return {
                "flagged": True,
                "reason": rule["reason"],
                "severity": rule["severity"],
            }

    return {"flagged": False, "reason": "No renal/hepatic contraindication detected", "severity": "low"}


def check_drug_drug_interactions(
    patient: dict[str, Any], medication: str
) -> dict[str, Any]:
    """
    Check for drug-drug interactions between the new medication and the patient's
    current medication list.

    Args:
        patient: dict with key ``current_medications`` (list[str]).
        medication: name of the new drug.

    Returns:
        Rule result dict with the most severe interaction found, or no-flag.
    """
    med = _normalize(medication)
    current = [_normalize(m) for m in patient.get("current_medications", [])]

    severity_order = {"high": 3, "medium": 2, "low": 1}
    worst: dict[str, Any] | None = None

    for existing_med in current:
        key = frozenset({med, existing_med})
        interaction = DDI_TABLE.get(key)
        if interaction:
            if worst is None or severity_order[interaction["severity"]] > severity_order[worst["severity"]]:
                worst = interaction

    if worst:
        return {
            "flagged": True,
            "reason": worst["reason"],
            "severity": worst["severity"],
        }

    return {"flagged": False, "reason": "No drug-drug interactions detected", "severity": "low"}


def check_age_contraindication(
    patient: dict[str, Any], medication: str
) -> dict[str, Any]:
    """
    Check age-based contraindications for the given medication.

    Args:
        patient: dict with key ``age`` (years).
        medication: drug name.

    Returns:
        Rule result dict.
    """
    med = _normalize(medication)
    age = patient.get("age")

    if age is None:
        return {"flagged": False, "reason": "Age not provided; age check skipped", "severity": "low"}

    rules = AGE_CONTRAINDICATIONS.get(med, [])
    for min_age, max_age, reason, severity in rules:
        min_ok = min_age is None or age > min_age
        max_ok = max_age is None or age <= max_age
        if min_ok and max_ok:
            return {"flagged": True, "reason": reason, "severity": severity}

    return {"flagged": False, "reason": "No age-based contraindication detected", "severity": "low"}


def run_all_rules(
    patient: dict[str, Any],
    medication: str,
    dose_mg: float | None = None,
) -> list[dict[str, Any]]:
    """
    Run all safety rules and return a list of rule results.

    Args:
        patient: patient profile dict.
        medication: name of the drug to evaluate.
        dose_mg: prescribed dose in milligrams (required for pediatric dosage check).

    Returns:
        List of rule result dicts, one per rule.
    """
    results = [
        check_allergy_contraindication(patient, medication),
        check_renal_hepatic_impairment(patient, medication),
        check_drug_drug_interactions(patient, medication),
        check_age_contraindication(patient, medication),
    ]
    if dose_mg is not None:
        results.append(check_pediatric_weight_dose(patient, medication, dose_mg))
    return results


def any_flagged(results: list[dict[str, Any]]) -> bool:
    """Return True if any rule result is flagged."""
    return any(r["flagged"] for r in results)
