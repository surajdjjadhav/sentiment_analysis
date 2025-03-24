import pandas as pd
import numpy as np
import os
import pickle
import json
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import logging
from src.Exception import MyException
import sys

# Ensure "reports" directory exists
os.makedirs("reports", exist_ok=True)

# Get DagsHub token from environment variables
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "surajdjjadhav"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Define Dagshub repository details
dagshub_uri = "https://dagshub.com"
repo_owner = "surajdjjadhav"
repo_name = "sentiment_analysis"

mlflow.set_tracking_uri(f"{dagshub_uri}/{repo_owner}/{repo_name}.mlflow")

# Function to load the trained model
def load_model(file_path: str):
    try:
        with open(file_path, "rb") as obj:
            model = pickle.load(obj)
            logging.info("Model loaded successfully")
            return model
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        raise MyException(e, sys)

# Function to load data
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        raise MyException(e, sys)

# Function to evaluate model
def model_evaluation(clf, x_test: np.array, y_test: np.array) -> dict:
    try:
        y_pred = clf.predict(x_test)
        metrics_dict = {
            "accuracy_score": accuracy_score(y_test, y_pred),
            "precision_score": precision_score(y_test, y_pred, average="macro"),  # Multi-class fix
            "recall_score": recall_score(y_test, y_pred, average="macro"),  # Multi-class fix
            "f1_score": f1_score(y_test, y_pred, average="macro"),  # Multi-class fix
        }
        logging.info(f"Model evaluation metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        raise MyException(e, sys)

# Function to save metrics
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, "w") as obj:
            json.dump(metrics, obj, indent=4)
        logging.info(f"Metrics saved to {file_path}")
    except Exception as e:
        raise MyException(e, sys)

# Function to save model info
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {"run_id": run_id, "file_path": model_path}
        with open(file_path, "w") as obj:
            json.dump(model_info, obj, indent=4)
        logging.info(f"Model info saved to {file_path}")
    except Exception as e:
        raise MyException(e, sys)

# Main function
def main():
    try:
        mlflow.set_experiment("my-dvc-pipeline")  # Ensure experiment exists

        with mlflow.start_run() as run:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            x_test = test_data.iloc[:, :-1]
            y_test = test_data.iloc[:, -1]

            metrics = model_evaluation(clf=clf, x_test=x_test, y_test=y_test)
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics in MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters
            if hasattr(clf, "get_params"):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log model
            mlflow.sklearn.log_model(clf, "model")
            save_model_info(run.info.run_id, "model", "reports/experiment_info.json")

            # Log metrics JSON file as an artifact
            mlflow.log_artifact("reports/metrics.json")
    
    except Exception as e:
        logging.error(f"Error in main(): {e}")  # Add logging before raising
        raise MyException(e, sys)

if __name__ == "__main__":
    main()
