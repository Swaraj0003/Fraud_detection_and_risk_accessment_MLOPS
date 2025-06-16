from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import mlflow
import mlflow.sklearn
from joblib import dump
from data_preprocessing import preprocessing
import os
import json

def model_training(data_path=r'data\loan_applications.csv'):
    x_scaled, y_encode = preprocessing(data_path)

    # 🔧 Set MLflow tracking URI to local folder
    mlflow.set_tracking_uri("file:///G:/data_science_projects/risk/Fraud_detection_and_risk_accessment_Mlops/mlruns")

    # 🎯 Set experiment name (it will be created if it doesn’t exist)
    mlflow.set_experiment("Fraud-Detection-Model")

    with mlflow.start_run(run_name="RandomForestClassifier"):

        # 📊 Data Split
        x_train, x_test, y_train, y_test = train_test_split(
            x_scaled, y_encode, test_size=0.2, random_state=42, stratify=y_encode
        )

        # ✅ Model Train
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(x_train, y_train)

        # 🧪 Predict
        y_pred = rf.predict(x_test)

        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)

        # 📋 Print
        print("\n🎯 Random Forest Classifier Results")
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        # 📝 Log parameters and metrics
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        model_dir = os.path.join("model", "models")
        os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

        model_path = os.path.join(model_dir, "random_forest_model.joblib")
        dump(rf, model_path)
        mlflow.sklearn.log_model(rf, artifact_path="random_forest_model")

        print(" Model logged to MLflow successfully.")
        metrics = {"accuracy": accuracy_score(y_test, y_pred),"f1_score": f1_score(y_test, y_pred)}
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

# Entry point
if __name__ == "__main__":
    model_training()
