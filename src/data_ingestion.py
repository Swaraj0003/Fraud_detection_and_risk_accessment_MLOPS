import pandas as pd

def ingestion(data_path):
    df=pd.read_csv(data_path)
    return df