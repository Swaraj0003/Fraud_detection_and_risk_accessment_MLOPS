from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os

# Load the saved scaler and model
scaler = joblib.load('artifacts/scaler.joblib')
model = joblib.load("models/random_forest_model.joblib")

# Load saved label encoders
label_encoders = {}
categorical_cols = [
    "loan_type",
    "purpose_of_loan",
    "employment_status",
    "property_ownership_status"
]

for col in categorical_cols:
    encoder_path = os.path.join("artifacts", f"{col}_encoder.joblib")
    if os.path.exists(encoder_path):
        label_encoders[col] = joblib.load(encoder_path)

# Initialize FastAPI app
app = FastAPI(title="Loan Risk Prediction App")

# Set Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Home page
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict_risk(
    request: Request,
    loan_type: str = Form(...),
    loan_amount_requested: float = Form(...),
    loan_tenure_months: int = Form(...),
    purpose_of_loan: str = Form(...),
    employment_status: str = Form(...),
    monthly_income: float = Form(...),
    cibil_score: int = Form(...),
    existing_emis_monthly: float = Form(...),
    debt_to_income_ratio: float = Form(...),
    property_ownership_status: str = Form(...),
    applicant_age: int = Form(...),
    number_of_dependents: int = Form(...)
):
    # Create input dataframe
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

    # Label encode categorical fields
    for col in categorical_cols:
        encoder = label_encoders.get(col)
        if encoder:
            try:
                input_data[col] = encoder.transform(input_data[col])
            except ValueError:
                return templates.TemplateResponse("form.html", {
                    "request": request,
                    "prediction": f"Invalid input value for {col.replace('_', ' ').title()}!"
                })

    # Scale numerical values
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]
    prediction_text = "High Risk" if prediction == 1 else "Low Risk"

    # Return result
    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": prediction_text
    })
