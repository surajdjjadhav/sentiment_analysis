from src.constens import CONFIG , VECTORIZER , ALGORITHUMS
import os
import pandas as pd 
import numpy as np
import dagshub 
import mlflow 
import mlflow.sklearn
import re
import string
import time 
pd.set_option("future.no_scilent_downcasting" , True)

import warnings

from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier 
from sklearn.metrics import f1_score , accuracy_score, recall_score  , precision_score 
from xgboost import XGBClassifier 
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
import scipy.sparse 

warnings.filterwarnings("ignore")


mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner = CONFIG[ "dagshub_repo_owner"]  , repo_name = CONFIG["dagshub_repo_name"]  , mlflow = True)
mlflow.set_experiment(CONFIG["expriment_name"])


def remove_stop_words(text):
    stopwords = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stopwords])

def remove_number(text):
    return " ".join(char for char in text if not char.isdigit())


def lower(text):
    return text.lower()

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in text.split())


def remove_puctuation(text):
    return re.sub(f"[{(re.escape(string.punctuation))}]" , " " , text)


def remove_url(text):
    return re.sub(r'https?://\s+|www\.\s+','',text)




def normalize_text(df):
    try:
        df["reviews"].apply(remove_stop_words)\
                     .apply(remove_number)\
                    .apply(lower)\
                    .apply(lemmatization)\
                    .apply(remove_puctuation)\
                    .apply(remove_url)\
                    .apply(normalize_text)
        return df
    except Exception as e:
        print(f" error occur at normalization{e}")




def load_data(file_path):
    df = pd.read_csv(file_path)
    df = normalize_text(df)
    df = df[df["sentiment"].isin(["positive" , "negative"])]
    df['sentiment'] =   df["sentiment"].map({"positive":1 , "negative":0 })
    return df



def train_evaluate(df):

    with mlflow.start_run(run_name = "all experiment") as parent_run :
        for algo_name , algo in ALGORITHUMS.items():
            for vec_name  , vec in VECTORIZER.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}" , nested = True) as child_run:
                    try:
                        x= vec.fit_transform(df["reviews"])
                        y = df["sentiment"]


                        x_train , x_test , y_train , y_test = train_test_split(x , y, randomstate = 42 , test_size = CONFIG["test_size"])


                        mlflow.log_params({
                            "vectorizer":vec ,
                            "Algorithum":algo,
                            "test_size": CONFIG["test_size"]
                        })

                        model = algo
                        model.fit(x_train , y_train)

                        log_model_params(algo_name , model)


                        y_pred = model.predict(x_test)

                        metrics = {
                            "accuracy": accuracy_score(y_train , y_pred),
                            "precision": precision_score(y_train , y_test),
                            "f1_score" : f1_score(y_train , y_pred),
                            "recall": recall_score(y_train , y_test)
                        }

                        mlflow.log_metrics(metrics)
                        input_example = x_test[:5] if not scipy.sparse.issparse(x_test) else x_test[:5].toarray()

                        mlflow.sklearn.log_model(model ,"model" , input_example = input_example)


                        print(f" algorithum {algo_name} vectorizer_name {vec_name}")
                        print(f"metrics {metrics}")
                    except Exception as e:
                        print(f" error in  training {algo_name} vectorizer{vec_name}  : {e}")
                        mlflow.log_params("error", str(e))


def log_model_params(algo_name , model):

    
    params_to_log = {}

    if algo_name == "LogisticRegression":
        params_to_log[["C"]] = model.C
    elif algo_name == "MultinomialNB":
        params_to_log["alpha"]= model.alpha
    elif algo_name == 'RandomForestClassifier':
        params_to_log["n_estimator"] = model.n_estimator
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == "XGBoost":
        params_to_log["n_estimator"] = model.n_estimator
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name== "GradinetBoosting":
        params_to_log["n_estimator"] = model.n_estimator
        params_to_log["learning_rate"]= model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)



if __name__ == "__main__":

    df = load_data(CONFIG["data_path"])
    train_evaluate(df)