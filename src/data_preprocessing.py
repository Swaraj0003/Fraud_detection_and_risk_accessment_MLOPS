import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_ingestion import ingestion
from joblib import dump
import joblib
import os
os.makedirs('artifacts', exist_ok=True)

def preprocessing(data_path):
    df = ingestion(data_path)

    # Drop unused or non-model features
    x = df.drop([
        'application_id', 'application_date', 'interest_rate_offered',
        'residential_address', 'gender', 'fraud_flag', 'loan_status',"customer_id", "fraud_type"
    ], axis=1)

    # Encode all categorical features (add any missing ones here)
    categorical_cols = x.select_dtypes(include='object').columns.tolist()

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])
        encoders[col] = le
        joblib.dump(le, f"artifacts/{col}_encoder.joblib")

    # Check if all columns are numeric now
    assert x.select_dtypes(include='object').shape[1] == 0, "Some columns are still strings!"

    # Feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    dump(scaler, 'artifacts/scaler.joblib')
    # Encode target
    y = df['loan_status']
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    y_encode = pd.Series(y_encoded)

    return x_scaled, y_encode
