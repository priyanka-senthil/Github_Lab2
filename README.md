# 🧠 GitHub Lab 2 – Automated Model Retraining & Evaluation (CI/CD Pipeline)

## ⚙️ Overview

This lab demonstrates a **complete MLOps CI/CD workflow** for retraining, evaluating, testing, and versioning a machine-learning model directly through **GitHub Actions**.

Each time code is pushed to the `main` branch (or triggered manually), the pipeline:

1. Cleans old artifacts
2. Retrains a Random Forest classifier on the Breast Cancer dataset
3. Evaluates the model and logs metrics
4. Runs automated tests
5. Commits new model and metrics back to the repository

---

## 🧩 Pipeline Highlights

**Workflow:** `.github/workflows/model_calibration_on_push.yml`

Key stages:
```bash
1️⃣ Checkout code
2️⃣ Install dependencies
3️⃣ Generate timestamp for artifacts
4️⃣ Clean old artifacts (models, metrics, data)
5️⃣ Retrain model → saves model_<timestamp>_rf_model.joblib
6️⃣ Evaluate model → creates metrics/<timestamp>_metrics.json
7️⃣ Run unit tests for training & evaluation
8️⃣ Commit & push new artifacts to GitHub
```

---

## 🧪 Major Code Enhancements

| Area                   | Original                                             | Modified / Improved                                                              |
| ---------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Dataset**            | Used `make_classification()` (random synthetic data) | Replaced with real `sklearn.datasets.load_breast_cancer()` dataset               |
| **Model Type**         | DecisionTreeClassifier                               | Upgraded to **RandomForestClassifier**                                           |
| **Timestamp Handling** | Hardcoded / manual                                   | Dynamic timestamp generation per run                                             |
| **File Organization**  | Artifacts scattered                                  | Standardized directories: `models/`, `metrics/`, `data/`                         |
| **Evaluation Script**  | Printed metrics only                                 | Saves clean JSON file: `metrics/<timestamp>_metrics.json`                        |
| **Testing**            | None                                                 | Added full **pytest** coverage: training + evaluation tests                      |
| **GitHub Actions**     | Single push trigger                                  | Added: cleanup, tests, re-scoped variables, manual trigger (`workflow_dispatch`) |
| **Artifact Hygiene**   | Old files persisted between runs                     | Added `Clean old artifacts` step to prevent shape mismatches                     |
| **Automation**         | Manual execution                                     | Full CI/CD retraining pipeline with commit automation                            |

---

## 🧪 Tests Added

**1️⃣ `test/test_model_training.py`**

* Verifies trained model exists
* Checks predictions match expected shape

**2️⃣ `test/test_evaluate_model.py`**

* Confirms metrics JSON file generation
* Validates metric keys (`accuracy`, `f1_score`)
* Ensures metric values are within 0–1 range
* Supports re-evaluation on existing models

---

## 🚀 Example CI/CD Run (Successful Log)
```bash
✅ Retrain Model
Timestamp received from GitHub Actions: 20251112185444
Model saved → models/model_20251112185444_rf_model.joblib
✅ Training complete.

✅ Evaluate Model and Log Metrics
Evaluating model for timestamp 20251112185444
✅ Metrics saved → metrics/20251112185444_metrics.json
✅ Verified metrics file.

✅ Run Unit Tests
All 4 tests passed in 1.04s ✅

✅ Commit & Push Changes
[main abc123] Add metrics and updated model
2 files changed, 3 insertions(+)
```

---

## 📁 Final Repository Structure
```
Github_Lab2/
├── .github/workflows/
│   ├── model_calibration_on_push.yml
│   └── model_calibration.yml
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── __init__.py
├── test/
│   ├── test_model_training.py
│   ├── test_evaluate_model.py
│   └── __init__.py
├── models/
├── metrics/
├── data/
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧠 How to Run Locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model manually
timestamp=$(date '+%Y%m%d%H%M%S')
python src/train_model.py --timestamp "$timestamp"

# 3. Evaluate model
python src/evaluate_model.py --timestamp "$timestamp"

# 4. Run tests
pytest -v test/
```

---

## ⚡ Triggering the Workflow

* **Automatic:** on every push to `main`
* **Manual:** via the "Run workflow" button under *Actions → Model Retraining on Push to Main*

---

## 🏁 Outcome

After each successful run:

* New model and metrics are saved in their respective folders
* JSON metrics are versioned
* Tests validate performance and structure
* Artifacts are committed automatically to the repo

✅ **Fully reproducible, tested, and version-controlled ML retraining workflow.**

---

**Author:** *Priyanka Senthil*  
**Updated:** November 2025  
**Environment:** Python 3.9 | scikit-learn | MLflow | GitHub Actions | Pytest

---