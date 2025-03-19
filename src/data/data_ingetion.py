import os
import pandas as pd
import numpy as np
import yaml
from src.connection.aws_config import ClientS3
from src.logger import logging
from src.Exception import MyException
from sklearn.model_selection import train_test_split
import sys

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Params received from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML error: %s", e)
        raise
    except Exception as e:
        raise MyException(e, sys)
    


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loaded successfully from: %s", data_url)
        return df
    except Exception as e:
        raise MyException(e, sys)



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df[df["sentiment"].isin(["positive", "negative"])]
        df["sentiment"] = df["sentiment"].replace({"positive": 1, "negative": 0})
        logging.info("Data preprocessing completed")
        return df
    except Exception as e:
        raise MyException(e, sys)
    


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        logging.info("%s directory created successfully", raw_data_path)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        logging.info("Train data saved successfully")

        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info("Test data saved successfully")
    except Exception as e:
        raise MyException(e, sys)
    


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params.get('data_ingestion', {}).get('test_size', 0.2)

        df = load_data(data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv')
        
        # s3 = s3_connection.s3_operations("bucket-name", "accesskey", "secretkey")
        # df = s3.fetch_file_from_s3("data.csv")


        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        raise MyException(e, sys)

if __name__ == '__main__':
    main()
    
