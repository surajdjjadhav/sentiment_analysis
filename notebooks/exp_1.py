import pandas as pd 
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re 
import string 
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download("omw-1.4")
nltk.download("wordnet")
nltk.download("punkt")

data = pd.read_csv("notebooks/data.csv")

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    return " ".join(words)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def lower_case(text):
    return text.lower()

def remove_punctuations(text):
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return re.sub('\s+', ' ', text).strip()

def remove_link(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df["review"] = df["reviews"].apply(lower_case)\
                                .apply(remove_stop_words)\
                                .apply(remove_numbers)\
                                .apply(remove_punctuations)\
                                .apply(remove_link)\
                                .apply(lemmatization)
    return df 

df = normalize_text(data)


def scaling(df):
    x = df["sentiment"].isin(["positive" , 'negative'])
    df= df[x]
    df["sentiment"]= df['sentiment'].map({'positive':1 , 'negative':0})
    return df

df = scaling(df=df)


vectorizer = CountVectorizer(max_features=50)
x=vectorizer.fit_transform(df["reviews"])
y= df['sentiment']

x_train , x_test , y_train , y_test = train_test_split(x , y, random_state = 42 , test_size = 0.20) 

import dagshub

mlflow.set_tracking_uri("https://dagshub.com/surajdjjadhav/sentiment_analysis.mlflow")

dagshub.init(repo_owner="surajdjjadhav", repo_name="sentiment_analysis", mlflow=True)

mlflow.set_experiment("Logistic Regression")

import os
import time 
import logging  

logging.basicConfig(level= logging.INFO , format = "%(asctime)s -%(levelname)s - %(message)s")

logging.info("starting mlflow run...")


with mlflow.start_run():
    start_time= time.time()
    try:
        logging.info("logging preprocessing parameters ...")
        mlflow.log_param("vectorizer", "Bag of Words")
        mlflow.log_param("num of features", 50)
        mlflow.log_param("test size", 0.20)


        logging.info("initilizaing Logistic Regression model")
        model= LogisticRegression(max_iter=1000)

        logging.info("fitting model")
        model.fit(x_train,y_train)

        logging.info("model training complete")


        logging.info("logging model parameter")
        mlflow.log_param("model" , "logistic_regression")
        

        logging.info("making_prediction")
        y_pred= model.predict(x_test)


        logging.info("avaluating model accurancy")

        accuracy= accuracy_score(y_test , y_pred)
        f1 = f1_score(y_test , y_pred)
        recall = recall_score(y_test , y_pred)
        precision = precision_score(y_test , y_pred)

        logging.info("logging evaluation metrix")
        mlflow.log_metric("accuracy score" , accuracy)
        mlflow.log_metric("precision" , precision)
        mlflow.log_metric("recall" , recall)
        mlflow.log_metric("f1 score " , f1)

        logging.info("saving and logging model")
        mlflow.sklearn.log_model(model , "model")
        end_time = time.time()

        logging.info(f" model training and model evaluation complited in {end_time - start_time :.2f} seconds")

        logging.info(f"accuracu : {accuracy}") 
        logging.info(f"fa score ; {f1}")
        logging.info(f"precision score { precision}")
        logging.info(f"recall score{recall}")

    except Exception as e:
        logging.error(f"error occured ar {e}" , exc_info = True)
