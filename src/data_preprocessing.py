import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.data_ingestion import data_ingestion

def preprocessing(data_path):
    df=data_ingestion(data_path)
    x = df.drop(['application_id', 'application_date', 'interest_rate_offered','residential_address','gender','fraud_flag','loan_status'], axis=1)
    encoders = {}
    for i in col:
        le = LabelEncoder()
        x_encoded = le.fit_transform(x[i])  # Use 'x' if that's your dataframe name
        encoders[i] = le
        x[i] = x_encoded 
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(x)   
    x_scaled=pd.DataFrame(x_scaled,columns=x.columns)
    y_encoded = le.fit_transform(y)
    y_encode=pd.Series(y_encoded)
    return x_scaled,y_encode