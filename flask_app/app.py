import os
import time
from flask_app.logger import logging
from flask import Flask, request, render_template
import pandas as pd
import pickle
import mlflow
import dagshub
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from src.data.data_preprocessing import normalize_text
from flask_app.Exception import MyException  
import sys



try:
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] =  "surajdjjadhav"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Define Dagshub repository details
    dagshub_uri = "https://dagshub.com"
    repo_owner = "surajdjjadhav"
    repo_name = "sentiment_analysis"

    mlflow.set_tracking_uri(f"{dagshub_uri}/{repo_owner}/{repo_name}.mlflow")
    # mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    # dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    
    logging.info("MLflow tracking URI set successfully.")
except Exception as e:
    logging.error(f"Failed to set MLflow tracking URI: {e}")
    raise MyException(e, sys)

app = Flask(__name__)

# Prometheus metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry)

model_name = "my_model"

def get_latest_model_version(model_name):
    try:
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])  # Fetch from Staging
        if not latest_version:
            latest_version = client.get_latest_versions(model_name, stages=["None"])  # Fallback to None
        return latest_version[0].version if latest_version else None
    except Exception as e:
        logging.error(f"Error retrieving model version: {e}")
        raise MyException(e, sys)

# Load the model and vectorizer
try:
    model_version = get_latest_model_version(model_name)
    if model_version is None:
        raise ValueError(f"No available version found for model '{model_name}'.")
    
    model_uri = f'models:/{model_name}/{model_version}'
    logging.info(f"Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    
    vectorizer_path = "model/vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        raise ValueError(f"Vectorizer file not found at {vectorizer_path}")
    
    vectorizer = pickle.load(open(vectorizer_path, 'rb'))
    logging.info("Model and vectorizer loaded successfully.")
except MyException as e:
    logging.error(f"Custom Exception: {e}")
    raise
except Exception as e:
    logging.error(f"Unexpected error loading model: {e}")
    raise MyException(e, sys)

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    try:
        response = render_template("index.html", result=None)
    except Exception as e:
        logging.error(f"Error rendering home page: {e}")
        response = "Internal Server Error", 500
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    
    try:
        text = request.form["text"]
        text = normalize_text(text)
        features = vectorizer.transform([text])
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
        result = model.predict(features_df)
        prediction = result[0]
        
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        logging.info(f"Prediction made: {prediction}")
        response = render_template("index.html", result=prediction)
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        response = "Prediction Error", 500
    
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return response

@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
    except Exception as e:
        logging.error(f"Error generating metrics: {e}")
        return "Metrics Error", 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
