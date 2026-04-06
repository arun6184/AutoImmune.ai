# AutoImmune.ai

A clinical decision-support web application that uses machine learning to detect autoimmune disorders from lab data and predict disease flares from clinical notes — built with Flask, scikit-learn, and a TF-IDF NLP pipeline.

---

## What it actually does

Two prediction engines under one authenticated web interface:

**1. Clinical Lab Classifier**
Takes 27 patient biomarkers (CBC panel, immunology, biochemistry) and runs them through a trained Random Forest model to classify the patient into one of five autoimmune diagnoses:
- Autoimmune Orchitis
- Graves' Disease
- Rheumatoid Arthritis
- Sjögren Syndrome
- Systemic Lupus Erythematosus (SLE)

**2. Flare Predictor (NLP)**
Accepts a `.txt` clinical note file, vectorizes it using TF-IDF, and runs it through a Multinomial Naive Bayes classifier to predict whether the patient is experiencing a disease flare or is in remission.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | Flask |
| Auth | SQLite + Werkzeug password hashing |
| Lab Classifier | Random Forest (scikit-learn) |
| NLP Flare Model | TF-IDF + Multinomial Naive Bayes (scikit-learn Pipeline) |
| Model Serialization | pickle / joblib |
| Frontend | Jinja2 templates, Inter font, Font Awesome |
| Data Generation | Custom synthetic clinical note generator |

---

## Project Structure

```
AutoImmune.ai/
├── app.py                  # Flask app — routes, auth, prediction endpoints
├── Main.py                 # Model training — Random Forest + KNN on Dataset.csv
├── clinical.py             # NLP pipeline training — TF-IDF + Naive Bayes on clinical notes
├── datacreate.py           # Synthetic clinical note generator (50 notes + labels.csv)
├── model.pickle            # Trained Random Forest classifier
├── flare_predictor.pkl     # Trained TF-IDF + NB pipeline
├── Dataset.csv             # Tabular patient data (27 features + Diagnosis label)
├── labels.csv              # Filename → flare/no flare mapping
├── clinical_notes/         # 50 synthetic .txt clinical notes
├── uploads/                # Runtime upload directory for user-submitted notes
├── templates/
│   ├── index.html          # Landing page
│   ├── login.html
│   ├── register.html
│   ├── home.html           # Dashboard
│   ├── clinical_predict.html
│   └── symptom_predict.html
└── users.db                # SQLite user store
```

---

## Getting Started

### Prerequisites

```bash
pip install flask scikit-learn numpy pandas joblib werkzeug tensorflow seaborn matplotlib
```

### Run the app

```bash
python app.py
```

Visit `http://127.0.0.1:5000` — register an account and you're in.

### Retrain the models (optional)

Train the lab classifier:
```bash
python Main.py
```

Regenerate synthetic notes and retrain the flare model:
```bash
python datacreate.py
python clinical.py
```

---

## How the models were built

### Random Forest (Lab Classifier)

`Main.py` loads `Dataset.csv`, label-encodes categorical columns (Gender, Diagnosis), splits 80/20, and trains a `RandomForestClassifier`. A KNN model is also trained for comparison. The RF model is serialized to `model.pickle`.

Input features fed to the model at inference time:

```
Patient ID, Age, Gender, Sickness Duration,
RBC, Hemoglobin, Hematocrit, MCV, MCH, MCHC, RDW, Reticulocyte Count,
WBC, Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils,
PLT Count, MPV,
ANA, Esbach, MBL Level, ESR, C3, C4, CRP
```

### TF-IDF + Naive Bayes (Flare Predictor)

`clinical.py` reads 50 clinical note `.txt` files, maps labels from `labels.csv` (flare=1, no flare=0), and trains a `make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())`. The pipeline is saved as `flare_predictor.pkl` and used directly on raw text at inference — no preprocessing needed at runtime.

---

## API / Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Landing page, redirects to dashboard if logged in |
| `/register` | GET, POST | User registration |
| `/login` | GET, POST | User login |
| `/logout` | GET | Clears session |
| `/home` | GET | Dashboard (auth required) |
| `/clinical` | GET | Clinical prediction form |
| `/predict_clinical` | POST | Runs RF model, returns diagnosis + recommendation |
| `/symptom_predict` | GET, POST | Upload `.txt` note, runs flare model |

---

## Notes for developers

- `users.db` is auto-created on first run via `init_db()` — no migration needed.
- The `uploads/` folder is created at startup with `os.makedirs(..., exist_ok=True)`.
- `no.py` is an earlier version of `app.py` without auth — kept for reference.
- The `secret_key` in `app.py` should be replaced with an environment variable before any deployment.
- Models are loaded once at startup, not per-request — keep that in mind for memory in production.

---

## Disclaimer

This tool is built for research and educational purposes. It is not a substitute for professional medical diagnosis or clinical judgment.
