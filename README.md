FRAUD DETECTION AND RISK ACCESSMENT MLOPS

This project aims to build an end-to-end machine learning pipeline for **loan fraud detection and risk assessment**, integrating MLOps best practices MLflow, model training and evaluation, CI/CD (GitHub Actions), and deployment on **Hugging Face Spaces** using FastAPI.

---

 Features

-  End-to-end ML pipeline using Scikit-learn
-  MLflow for model tracking , CI/CD with GitHub Actions (linting, testing, reproducibility)
-  FastAPI-based deployment
-  Auto-deployed on Hugging Face Spaces
-  Label encoding, scaling, metrics logging
-  Modular source code in `src/`

---
              |

---

 Project Structure
 
├── app.py 

├── requirements.txt

├── models

├── artifacts

├── data
 
├── src/

│ ├── data_ingestion.py

│ ├── data_preprocessing.py

│ └── model_training.py

├── .github/workflows

└── README.md

---

 Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/Swaraj0003/Fraud_detection_and_risk_accessment_Mlops.git
cd Fraud_detection_and_risk_accessment_Mlops
Install dependencies

pip install -r requirements.txt

Start the FastAPI app (local)

uvicorn main:app --reload

