
CONFIG = {
    "data_path":"notebooks\data.csv",
    "test_size":0.20,
    "mlflow_tracking_uri":  "https://dagshub.com/surajdjjadhav/sentiment_analysis.mlflow",
    "dagshub_repo_owner": 'surajdjjadhav',
    "dagshub_repo_name" : 'sentiment_analysis' ,
    "expriment_nmae" : "Bow vs Tfidf"
}

VECTORIZER = {
    "Bow": CountVectorizer(),
    "TFI_DF" :  TfidfVectorizer()
}


ALGORITHUMS = {
    "LogesticRegression": LogesticRegression(),
    "XGBoost" : XGBoost(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting" : GradientBoostingClassifier(),
    "multinomialNB" : MultinomialNB()
}