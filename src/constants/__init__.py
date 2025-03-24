import os 


CONFIG = {
    "data_path":"notebooks\data.csv",
    "test_size":0.20,
    "mlflow_tracking_uri":  os.getenv("mlflow_tracking_uri"),
    "dagshub_repo_owner": 'surajdjjadhav',
    "dagshub_repo_name" : 'sentiment_analysis' ,
    "experiment_name" : "Bow_vs_Tfidf_v"
}

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")