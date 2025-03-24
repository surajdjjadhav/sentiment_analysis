import pandas as pd 
from src.logger import logging
from src.Exception import MyException
import sys




def load_data(file_path):
    """this function is useful for data loading from given file path"""

    try :
        df = pd.read_csv(file_path)
        logging.info(f"data loaded sucessfuly : {file_path}")

        return df
    except FileNotFoundError:
        logging.error(f"file not found  in path : { file_path}") 
        raise
    except Exception as e :
        raise MyException(e , sys)