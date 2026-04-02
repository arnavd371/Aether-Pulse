# Aether-Pulse: Clinical Decision Support System (CDSS)

Aether Pulse is a hybrid rule based + machine-learning medication safety checker
designed to assist clinicians in identifying potentially dangerous prescriptions
before they reach the patient.


## Architecture Overview

```
Patient Profile + Medication
          │
          ▼
┌─────────────────────┐
│  Rules Engine       │  ← deterministic, always runs first
│  safety_rules.py    │     • Allergy contraindications
│                     │     • Pediatric weight-based dosage
│                     │     • Renal / hepatic impairment
│                     │     • Drug-drug interactions
│                     │     • Age-based contraindications
└────────┬────────────┘
         │ any flag? → return immediately (source = "rules")
         │ no flags?
         ▼
┌─────────────────────┐
│  ML Model           │  ← XGBoost binary classifier
│  (XGBClassifier)    │     • confidence ≥ threshold → source = "ml"
│                     │     • confidence < threshold → source = "fallback"
└────────┬────────────┘
         │
         ▼
  Unified Decision
  { safe, flags, confidence, source }
```

---

## Project Structure

```
Aether-Pulse/
├── src/
│   ├── rules_engine/
│   │   └── safety_rules.py       # Deterministic safety rules
│   ├── ml_model/
│   │   ├── preprocessing.py      # Feature engineering pipeline
│   │   ├── train.py              # XGBoost training script
│   │   └── evaluate.py           # Blind validation & metrics
│   ├── hybrid_engine/
│   │   └── decision_engine.py    # Hybrid rules + ML engine
│   └── api/
│       └── app.py                # FastAPI REST API
├── notebooks/
│   └── aether_pulse_training.ipynb  # Google Colab training notebook
├── requirements.txt
├── .gitignore
└── README.md
```



## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/arnavd371/Aether-Pulse.git
cd Aether-Pulse
```

### 2. Create a virtual environment (Python 3.10+)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```


## Training (Google Colab)

Open the training notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arnavd371/Aether-Pulse/blob/main/notebooks/aether_pulse_training.ipynb)

> **Note:** No dataset is committed to this repository. All data must be loaded
> at runtime from Google Drive or HuggingFace Datasets. Set the configuration
> variables in Cell 0 of the notebook before running.

### Training from the command line

```bash
python src/ml_model/train.py \
    --data_path /path/to/medication_safety.csv \
    --output_path model.pkl \
    --tuner grid
```

Environment variable alternative:

```bash
export AETHER_DATA_PATH=/path/to/medication_safety.csv
python src/ml_model/train.py
```

### Evaluation

```bash
python src/ml_model/evaluate.py \
    --data_path /path/to/test_data.csv \
    --model_path model.pkl \
    --output_dir eval_outputs/
```



## Running the API

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Set the model path via environment variable:

```bash
export AETHER_MODEL_PATH=/path/to/model.pkl
uvicorn src.api.app:app --reload
```



## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Check medication safety

```bash
curl -X POST http://localhost:8000/check-medication \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {
      "age": 7,
      "weight_kg": 22,
      "sex": "M",
      "diagnosis_codes": ["J06", "R50"],
      "allergies": [],
      "current_medications": [],
      "creatinine": 0.6,
      "alt": 18
    },
    "medication": "aspirin",
    "dose_mg": 500
  }'
```

**Response:**

```json
{
  "safe": false,
  "flags": [
    {
      "flagged": true,
      "reason": "Aspirin in children <18 years: risk of Reye's syndrome",
      "severity": "high"
    }
  ],
  "confidence": 1.0,
  "source": "rules"
}
```

---

## Key Design Constraints

- **No datasets committed** — all data loaded at runtime from Google Drive or HuggingFace.
- **HIPAA-conscious design** — no raw patient data is persisted; all inputs are treated as ephemeral.
- **`model.pkl` excluded from git** via `.gitignore`.
- **Google Colab compatible** — no local-only dependencies; relative paths used throughout.
- **Python 3.10+** required.



## Team

| Role | Name |
|------|------|
| Developer | Muhammad Hamza Shafoat |
| Developer | Tanisha Jain |
| Developer | Brian Liu |
| Developer | Arnav Dhiman |
| Developer | Jackson Raines |
| Developer | Basmala El-boudy |
| **Mentor** | **Ben Amoakoh** |

---

## License

This project is for educational and research purposes. See `LICENSE` for details.
