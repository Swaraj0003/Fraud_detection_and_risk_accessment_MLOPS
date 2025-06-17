from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from joblib import load


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump
import os

# Load the saved scaler
scaler = load('artifacts/scaler.joblib')
# Load model
model = joblib.load("models/model.joblib")

# Init FastAPI app
app = FastAPI(title="Loan Risk Prediction App")

# Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Home page with HTML form
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Handle form submission
@app.post("/predict", response_class=HTMLResponse)
async def predict_risk(
    request: Request,
    loan_type: int = Form(...),
    loan_amount_requested: float = Form(...),
    loan_tenure_months: int = Form(...),
    purpose_of_loan: int = Form(...),
    employment_status: int = Form(...),
    monthly_income: float = Form(...),
    cibil_score: int = Form(...),
    existing_emis_monthly: float = Form(...),
    debt_to_income_ratio: float = Form(...),
    property_ownership_status: int = Form(...),
    applicant_age: int = Form(...),
    number_of_dependents: int = Form(...)
):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        "loan_type": loan_type,
        "loan_amount_requested": loan_amount_requested,
        "loan_tenure_months": loan_tenure_months,
        "purpose_of_loan": purpose_of_loan,
        "employment_status": employment_status,
        "monthly_income": monthly_income,
        "cibil_score": cibil_score,
        "existing_emis_monthly": existing_emis_monthly,
        "debt_to_income_ratio": debt_to_income_ratio,
        "property_ownership_status": property_ownership_status,
        "applicant_age": applicant_age,
        "number_of_dependents": number_of_dependents
    }])
    
     def preprocess_and_save(data: input_data, save_dir: str = "artifacts"):
    

    

    # Copy to avoid modifying original
    

    # Identify categorical columns
    categorical_cols = x.select_dtypes(include='object').columns.tolist()

    encoders = {}

    # Label encoding
    for col in categorical_cols:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
        encoders[col] = le

        # Save encoder
        dump(le, os.path.join(save_dir, f"{col}_encoder.joblib"))

    # Sanity check
    assert x.select_dtypes(include='object').shape[1] == 0, "Some columns are still strings!"

    # Feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

    # Save scaler
    dump(scaler, os.path.join(save_dir, "scaler.joblib"))

    return x_scaled

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_text = "High Risk" if prediction == 1 else "Low Risk"

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction_text
    })
