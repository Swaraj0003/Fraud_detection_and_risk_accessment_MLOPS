# 🛡️ Fraud Detection and Risk Assessment MLOps Project

This project aims to build an end-to-end machine learning pipeline for **loan fraud detection and risk assessment**, integrating MLOps best practices MLflow, model training and evaluation, CI/CD (GitHub Actions), and deployment on **Hugging Face Spaces** using FastAPI.

---

## 🚀 Features

- ✅ End-to-end ML pipeline using Scikit-learn
- ✅ MLflow for model tracking , CI/CD with GitHub Actions (linting, testing, reproducibility)
- ✅ FastAPI-based deployment
- ✅ Auto-deployed on Hugging Face Spaces
- ✅ Label encoding, scaling, metrics logging
- ✅ Modular source code in `src/`

---

## 🔧 Tech Stack

| Tool            | Purpose                                |
|-----------------|----------------------------------------|
| `FastAPI`       | API for model inference                |
| `Scikit-learn`  | ML model training                      |
| `GitHub Actions`| CI/CD automation                       |
| `flake8`        | Code style linting                     |                   |

---

## 🗂️ Project Structure
├── app.py / main.py # FastAPI app
├── requirements.txt
├── models/ # Saved ML models
├── artifacts/ # Scaler & encoders
├── data/
│ ├── raw/ # Original data
│ 
├── src/
│ ├── data_ingestion.py
│ ├── data_preprocessing.py
│ └── model_training.py
├── tests/ # Test case
├── .github/workflows/ # GitHub CI workflows
└── README.md
---

## 🛠️ Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/Swaraj0003/Fraud_detection_and_risk_accessment_Mlops.git
cd Fraud_detection_and_risk_accessment_Mlops
Install dependencies

pip install -r requirements.txt

Start the FastAPI app (local)

uvicorn main:app --reload

