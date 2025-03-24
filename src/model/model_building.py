import pandas as pd
import numpy as np 
import yaml
from sklearn.linear_model import LogisticRegression
import pickle 
from src.logger import logging
from src.Exception  import MyException
import sys

from src.utils.utils import load_data
""" model building / fitting model """



def train_model(x_train:np.ndarray , y_train:np.ndarray)-> LogisticRegression:
    """train logistic regression model"""

    try:
        model = LogisticRegression(C=1, solver= "liblinear" , penalty="l1")

        model.fit(x_train,y_train) 
        logging.info("model training sucessfuly")
        return model
    except Exception as e:
        raise MyException(e,sys)


def save_model(model , file_path):
    try:
        with open(file_path , "wb") as obj:
            pickle.dump(model  , obj)
        logging.info( "model saved sucessfuly")
    except Exception as e:
        raise MyException(e , sys)
    

def main():
    try:
        data = load_data('./data/processed/train_bow.csv')
        x_train= data.iloc[: , :-1]
        y_train = data.iloc[: ,-1]
        model = train_model(x_train , y_train )

        save_model(model , "models/model.pkl")
    except Exception as e:
        raise MyException(e , sys) 
    

if __name__ == "__main__":
    main()

