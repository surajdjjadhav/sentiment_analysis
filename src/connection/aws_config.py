from boto3 import client
import sys
from src.logger import logging
from src.Exception import MyException

import pandas as pd 
from io import StringIO
class ClientS3():

    def __init__(self , aws_acces_key , region_name , aws_secerate_acess_key , bucket_name):
        self.access_key = aws_acces_key
        self.secerate_key = aws_secerate_acess_key
        self.region_name = region_name 
        self.bucket_name = bucket_name
    
    def connection(self):
        try:
            connection = client(
                aws_access_key = self.access_key,
                aws_secerate_access_key = self.secerate_key,
                region_name = self.region_name

            )

            logging.info("aws connection sucessefuly done")
            return connection
        except Exception as e:
            logging.error(f"error occuring during connection ar {e}")
            raise MyException(e , sys)
    
    def load_data(self , file_key):
        try:
            conn = self.connection()
            obj = conn.get_object(bucket_name = self.bucket_name , key = file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"data loading sucessfuly form awss3 bucket {self.bucket_name}")
            return df 
        except Exception as e:
            logging.error(f"error occur durning loading data form aws at {e}")
            raise MyException(e , sys)