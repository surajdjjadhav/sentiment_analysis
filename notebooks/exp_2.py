from src.constants import CONFIG
import os
import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
import re
import string
from src.logger import logging
from src.Exception import MyException
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from xgboost import XGBClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse
import warnings
import sys

warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')
logging.info("nltk libraries downloading done")

try:
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)

    mlflow.set_experiment(CONFIG["experiment_name"])
    logging.info("MLflow and Dagshub initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing MLflow and Dagshub: {e}")
    raise MyException(f"Initialization error: {e}")

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
VECTORIZER = {
    "Bow": CountVectorizer(),
    "TFIDF": TfidfVectorizer()
}

ALGORITHMS = {
    "LogisticRegression": LogisticRegression(),
    "XGBoost": XGBClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "MultinomialNB": MultinomialNB()
}

def train_evaluate(df):
    try:
        with mlflow.start_run(run_name="all_experiment") as parent_run:
            for algo_name, algo in ALGORITHMS.items():
                for vec_name, vec in VECTORIZER.items():
                    with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                        try:
                            logging.info(f"Training {algo_name} with {vec_name}")
                            x = vec.fit_transform(df["reviews"])
                            y = df["sentiment"]
                            x_train, x_test, y_train, y_test = train_test_split(
                                x, y, random_state=42, test_size=CONFIG["test_size"])
                            
                            mlflow.log_params({
                                "vectorizer": vec_name,
                                "algorithm": algo_name,
                                "test_size": CONFIG["test_size"]
                            })
                            
                            model = algo
                            model.fit(x_train, y_train)
                            log_model_params(algo_name, model)
                            
                            y_pred = model.predict(x_test)
                            metrics = {
                                "accuracy": accuracy_score(y_test, y_pred),
                                "precision": precision_score(y_test, y_pred),
                                "f1_score": f1_score(y_test, y_pred),
                                "recall": recall_score(y_test, y_pred)
                            }
                            mlflow.log_metrics(metrics)
                            input_example = x_test[:5] if not scipy.sparse.issparse(x_test) else x_test[:5].toarray()
                            mlflow.sklearn.log_model(model, "model", input_example=input_example)
                            
                            logging.info(f"Model trained: {algo_name} with {vec_name} | Metrics: {metrics}")
                        except Exception as e:
                            logging.error(f"Error in training {algo_name} with {vec_name}: {e}")
                            mlflow.log_param("error", str(e))
                            raise MyException(e, sys)
    except Exception as e:
        logging.error(f"Error in train_evaluate function: {e}")
        raise MyException(e, sys)

def log_model_params(algo_name, model):
    try:
        params_to_log = {}
        if algo_name == "LogisticRegression":
            params_to_log["C"] = model.C
        elif algo_name == "MultinomialNB":
            params_to_log["alpha"] = model.alpha
        elif algo_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
            params_to_log.update({
                "n_estimators": model.n_estimators,
                "learning_rate": getattr(model, "learning_rate", None),
                "max_depth": getattr(model, "max_depth", None)
            })
        mlflow.log_params(params_to_log)
    except Exception as e:
        logging.error(f"Error in logging model parameters: {e}")
        raise MyException(e, sys)

if __name__ == "__main__":
    try:
        df = load_data(CONFIG["data_path"])
        print(df.head())
        train_evaluate(df)
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise MyException(e, sys)
