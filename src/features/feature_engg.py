import pandas as pd
import numpy as np
import os
import sys
import yaml
import pickle
from src.logger import logging
from src.Exception import MyException  # Fix incorrect import

from sklearn.feature_extraction.text import CountVectorizer


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df.fillna(" ", inplace=True)
        
        # Validate required columns
        required_columns = {"review", "sentiment"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise MyException(f"Missing required columns in {file_path}: {missing_columns}", sys)

        logging.info(f"Data loaded and null values filled from {file_path}")
        return df
    except Exception as e:
        logging.error("Failed to load CSV file", exc_info=True)
        raise MyException(e, sys)


def load_params(file_path: str) -> dict:
    try:
        with open(file_path, "r") as obj:
            params = yaml.safe_load(obj)
            logging.info(f"Parameters retrieved from {file_path}")
            return params
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise MyException(f"File not found: {file_path}", sys)
    except yaml.YAMLError as e:
        logging.error(f"YAML error: {e}", exc_info=True)
        raise MyException(e, sys)
    except Exception as e:
        raise MyException(e, sys)


def apply_BOW(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        logging.info("Applying Bag of Words...")

        vectorizer = CountVectorizer(max_features=max_features)  # Set max features
        x_train = train_data["review"].values
        y_train = train_data["sentiment"].values
        x_test = test_data["review"].values
        y_test = test_data["sentiment"].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)  # Fix: use transform

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test

        os.makedirs("model", exist_ok=True)
        pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

        logging.info("Bag of Words applied and data transformed successfully.")

        return train_df, test_df  # Fix: return the transformed data

    except Exception as e:
        logging.error("Error while applying Bag of Words", exc_info=True)
        raise MyException(e, sys)


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save DataFrame to CSV file."""
    try:
        dir_name = os.path.dirname(file_path) or "."
        os.makedirs(dir_name, exist_ok=True)

        df.to_csv(file_path, index=False)
        logging.info(f"Data saved to: {file_path}")
    except Exception as e:
        logging.error("Failed to save data", exc_info=True)
        raise MyException(e, sys)


def main():
    try:
        params = load_params("params.yaml")
        max_features = params.get("feature_engineering", {}).get("max_features", 1000)  # Fix: safer key access

        train_data = load_data("./data/interim/train_process_data.csv")
        test_data = load_data("./data/interim/test_processed_data.csv")

        train_df, test_df = apply_BOW(train_data, test_data, max_features)  # Fix: get return values

        save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))

    except Exception as e:
        logging.error("Error in main execution", exc_info=True)
        raise MyException(e, sys)


if __name__ == "__main__":
    main()
