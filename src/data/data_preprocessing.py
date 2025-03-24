import pandas as pd
import numpy as np
import nltk
import os
import string
import re
import sys
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from src.logger import logging
from src.Exception import MyException

# Ensure required NLTK data is available
nltk.download("wordnet")
nltk.download("stopwords")



def normalize_text(input_data, col="text"):
    try:
        logging.info("Text normalization initialized")
        
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        def preprocess_text(text):
            try:
                text = str(text)  # Ensure text is a string
                text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
                text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
                text = re.sub(r'\d+', '', text)  # Remove numbers
                text = text.lower()  # Convert to lowercase
                text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
                text = " ".join(lemmatizer.lemmatize(word) for word in text.split())  # Lemmatization
                return text
            except Exception as e:
                raise MyException(e, sys)
        
        # If input is a DataFrame, apply preprocessing to the specified column
        if isinstance(input_data, pd.DataFrame):
            if col not in input_data.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            return input_data[col].apply(preprocess_text)

        # If input is a string, process it directly
        elif isinstance(input_data, str):
            return preprocess_text(input_data)

        else:
            raise TypeError("Input must be either a string or a DataFrame with a text column")

    except Exception as e:
        logging.error(f"Error in text normalization: {e}")
        raise MyException(e, sys)

def main():
    try:
        train_df = pd.read_csv("./data/raw/train.csv")
        test_df = pd.read_csv("./data/raw/test.csv")

        train_processed_data = normalize_text(train_df , "review")
        test_processed_data = normalize_text(test_df , "review")


        data_path = os.path.join("./data" , "interim")
        os.makedirs(data_path , exist_ok = True)
        logging.info(f"directory created sucessfuly for saving preprocessed data")


        train_processed_data.to_csv(os.path.join(data_path , "train_process_data.csv"), index = False)
        test_processed_data.to_csv(os.path.join(data_path , "test_processed_data.csv") , index = False) 
        logging.info(f"processed data saved to : {data_path}")  
    except Exception as e:
        raise MyException(e , sys)
    


if __name__ == "__main__" :
    main()