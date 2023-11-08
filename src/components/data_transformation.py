import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from geopy.distance import geodesic

@dataclass
class DataTransformationConfig:
   preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
   def __init__(self):
      self.transformation_config = DataTransformationConfig()

   def get_data_transformation_object(self):
      try:
         logging.info('Data Transformation Initiated')

         #segregating Categorical and Numerical columns
         Numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
            'multiple_deliveries', 'Distance']
         categorical_columns = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
            'Type_of_vehicle', 'Festival', 'City']
         
         logging.info('Pipeline Initiated')

         #Numerical Pipeline

         num_pipeline = Pipeline(
         steps=[
            ('impute',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())
         ])

         #categorical pipeline

         cat_pipeline = Pipeline(
            steps=[
               ('impute',SimpleImputer(strategy='most_frequent')),
               ('oneHot',OneHotEncoder(sparse_output=False)),
               ('scaler',StandardScaler())
            ]
         )

         #Combine Num and Cat using columntransformer
         preprocessor = ColumnTransformer(
            [('num_pipeline',num_pipeline,Numerical_columns),
            ('cat_pipeline',cat_pipeline,categorical_columns)]
         )

         return preprocessor

      except Exception as e:
         logging.info('Exception occured at Data Transformation stage')
         raise CustomException(e,sys)
   
   def initiate_data_transformation(self,train_path,test_path):
      try:
         #reading traina and test data

         train_df = pd.read_csv(train_path)
         test_df = pd.read_csv(test_path)
         logging.info('Reading train and test data completed')

         #calculate the distance of each pair using geodesic library
         train_df['Distance'] = np.nan
         test_df['Distance'] = np.nan
      
         for i in range(len(train_df)):
             train_df.loc[i,'Distance'] = geodesic((train_df.loc[i,'Restaurant_latitude'],
                                    train_df.loc[i,'Restaurant_longitude']),
                                    (train_df.loc[i,'Delivery_location_latitude'],
                                    train_df.loc[i,'Delivery_location_longitude'])).km
             
         for i in range(len(test_df)):
             test_df.loc[i,'Distance'] = geodesic((test_df.loc[i,'Restaurant_latitude'],
                                    test_df.loc[i,'Restaurant_longitude']),
                                    (test_df.loc[i,'Delivery_location_latitude'],
                                    test_df.loc[i,'Delivery_location_longitude'])).km
             
         logging.info('Calculating Distance is completed')
         logging.info(f"Train DataFrame Head: {train_df.head().to_string()}")
         logging.info(f"Test DataFrame Head: {test_df.head().to_string()}")

         logging.info('Obtaining Prepeocessing object ')

         preprocessor_obj = self.get_data_transformation_object()

         target_column = 'Time_taken_min'
         drop_columns =[target_column,'ID','Delivery_person_ID','Restaurant_latitude',
                       'Restaurant_longitude','Delivery_location_latitude',
                         'Delivery_location_longitude','Order_Date','Time_Orderd',
                         'Time_Order_picked']
         
         input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
         target_feature_train_df = train_df[target_column]

         input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
         target_feature_test_df = test_df[target_column]

         #Transforming using preprocessor obj
         input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
         input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

         logging.info('Applying preprocessing object on training and testing datasets')

         train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
         test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

         save_object(
            file_path = self.transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
         )
         logging.info('Preprocessor pickle file saved')

         return(
            train_arr,
            test_arr,
            self.transformation_config.preprocessor_obj_file_path
         )

      except Exception as e:
         logging.info('Exception occured at initiatede data transformation')
         raise CustomException(e,sys)