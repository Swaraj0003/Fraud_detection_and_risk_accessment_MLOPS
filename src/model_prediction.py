from joblib import dump, load
from src.model_evaluation import evaluation_model

def prediction(data_path):
    loaded_model = load('random_forest_model.joblib')
    y_pred_rf = loaded_model.predict(x_test)
    evaluate_model(y_test, y_pred_rf, "Random Forest")