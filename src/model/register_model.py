import json
import mlflow
import sys
import dagshub
from src.logger import logging
from src.Exception import MyException
import warnings
import os

warnings.filterwarnings("ignore")

# dagshub_token = os.getenv("capstone_test")

# if not dagshub_token:
#     raise EnvironmentError("capstone_test environment variable not set")

# os.environ["mlflow_tracking_username"] = dagshub_token
# os.environ["mlflow_tracking_password"] = dagshub_token

repo_owner = "surajdjjadhav"
repo_name = "sentiment_analysis"

"""this is for local use"""
mlflow.set_tracking_uri("https://dagshub.com/surajdjjadhav/sentiment_analysis.mlflow")

dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as obj:
            model_info = json.load(obj)

        logging.info(f"Model info loaded from {file_path}")
        return model_info

    except FileNotFoundError:
        logging.error(f"File not found at path: {file_path}")
        raise
    except Exception as e:
        raise MyException(e, sys)


def register_model(model_name: str, model_info: dict):
    """Register model to MLflow model registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['file_path']}"

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.info(f"Model {model_name} version {model_version.version} registered and transitioned to Staging")
    except Exception as e:
        raise MyException(e, sys)



def main():
    try:
        model_info_path = "reports/experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error(f"Failed to complete model registration: {e}")
        raise MyException(e, sys)


if __name__ == "__main__":
    main()
