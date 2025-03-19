import pandas as pd
from src.constants import CONFIG  # Ensure correct import
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re
import sys
from src.logger import logging
from src.Exception import MyException

# Download required nltk resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize MLflow and DagsHub
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
logging.info("MLflow tracking URI setup done for exp_3")

dagshub.init(repo_name=CONFIG["dagshub_repo_name"], repo_owner=CONFIG["dagshub_repo_owner"], mlflow=True)
logging.info("DagsHub initialization done for exp_3")

# Set experiment safely
mlflow.set_experiment("final_best_params_v2")
logging.info("MLflow experiment setup done for exp_3")

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        return " ".join([word for word in text.split() if word not in stop_words])
    except Exception as e:
        raise MyException(e, sys)

def remove_number(text):
    try:
        return re.sub(r'\d+', '', text)
    except Exception as e:
        raise MyException(e, sys)

def lower(text):
    try:
        return text.lower()
    except Exception as e:
        raise MyException(e, sys)
    
def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        return " ".join(lemmatizer.lemmatize(word) for word in text.split())
    except Exception as e:
        raise MyException(e, sys)
    
def remove_punctuation(text):
    try:
        return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    except Exception as e:
        raise MyException(e, sys)

def remove_url(text):
    try:
        return re.sub(r'https?://\S+|www\.\S+', '', text)
    except Exception as e:
        raise MyException(e, sys)

def normalize_text(df):
    try:
        logging.info("Text normalization initialized")
        df["reviews"] = df["reviews"].apply(lambda x: lower(x))\
                                    .apply(lambda x: remove_punctuation(x))\
                                    .apply(lambda x: remove_stop_words(x))\
                                    .apply(lambda x: remove_number(x))\
                                    .apply(lambda x: lemmatization(x))\
                                    .apply(lambda x: remove_url(x))
        logging.info("Text normalization completed")
        return df
    except Exception as e:
        raise MyException(e, sys)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loading completed")

        df = normalize_text(df)

        df = df[df["sentiment"].isin(["positive", "negative"])]
        df['sentiment'] = df["sentiment"].map({"positive": 1, "negative": 0})
        logging.info("Target column encoding completed")
        return df
    except Exception as e:
        raise MyException(e, sys)

def train_log_model(df):
    try:
        logging.info("Starting model training...")
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df["reviews"])
        y = df["sentiment"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=CONFIG["test_size"])
        logging.info("Data split into training and testing sets")

        params_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"]
        }

        with mlflow.start_run():
            logging.info("Starting GridSearchCV for hyperparameter tuning")
            gridsearch = GridSearchCV(LogisticRegression(), params_grid, cv=5, scoring="f1", n_jobs=-1)
            gridsearch.fit(X_train, y_train)
            logging.info("GridSearchCV completed")

            for param, mean_score, std_score in zip(
                    gridsearch.cv_results_["params"],
                    gridsearch.cv_results_["mean_test_score"],
                    gridsearch.cv_results_["std_test_score"]):
                with mlflow.start_run(run_name=f"LR with params {param}", nested=True):
                    logging.info(f"Training model with params: {param}")
                    model = LogisticRegression(**param)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "f1_score": f1_score(y_test, y_pred),
                        "precision_score": precision_score(y_test, y_pred),
                        "recall_score": recall_score(y_test, y_pred)
                    }
                    mlflow.log_params(param)
                    mlflow.log_metrics(metrics)
                    logging.info(f"Metrics logged for params {param}: {metrics}")

            best_params = gridsearch.best_params_
            best_model = gridsearch.best_estimator_
            best_f1 = gridsearch.best_score_
            logging.info(f"Best parameters found: {best_params} with F1 score {best_f1}")

            mlflow.log_params(best_params)
            mlflow.log_metric("best_score", best_f1)
            mlflow.sklearn.log_model(best_model, "model")

            logging.info("Best model logged successfully")
            print(f"Best params: {best_params} | Best F1 score: {best_f1}")
    except Exception as e:
        logging.error("Error occurred in model training", exc_info=True)
        raise MyException(e, sys)
    
if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_log_model(df)
