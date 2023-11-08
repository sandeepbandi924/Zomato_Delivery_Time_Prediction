import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation,DataTransformationConfig

@dataclass
class DataIngestionConfig:
   train_data_path = os.path.join('artifacts','train.csv')
   test_data_path = os.path.join('artifacts','test.csv')
   raw_data_path = os.path.join('artifacts','raw.csv')

class DataIngestion:

   def __init__(self):
      self.ingestion_config = DataIngestionConfig()

   def initiate_data_ingestion(self):
      try:
      
         logging.info('Data Ingestion Method starts')
         #read dataset
         df = pd.read_csv(os.path.join('notebooks/data','zomato.csv'))
         logging.info('Dataset raed as pandas dataframe')

         os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
         df.to_csv(self.ingestion_config.raw_data_path,index=False)

         logging.info('Train Test Split')
         train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)

         train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
         test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
         logging.info('Ingestion of the data is completed')

         return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
         )


      except Exception as e:
         logging.info('Exception occured at Data Ingestion stage')
         raise CustomException(e,sys)
      

if __name__ == '__main__':
   ingestion = DataIngestion()
   train_data,test_data =ingestion.initiate_data_ingestion()
   transform = DataTransformation()
   transform.initiate_data_transformation(train_data,test_data)