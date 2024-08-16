import pandas as pd
import numpy as np
from src.dimond_price_prediction.logger.logging import logging
from src.dimond_price_prediction.exception.exception import CustomException
from src.dimond_price_prediction.utils.utils import save_object

import os 
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()


    def get_data_tranformation(self):
        try:
            cat_cols = ['cut','color','clarity']
            num_cols = ['carat','depth','table', 'x','y','z']

            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline initated...")

            num_pipeline = Pipeline(
            steps=[
                ("imputer",SimpleImputer()),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder", OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ("scaler",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                    ("num_pipeline", num_pipeline, num_cols),
                    ("cat_pipeline", cat_pipeline, cat_cols)
            ])
            return preprocessor

        except Exception as e:
            logging.info("Exception during data tranformation")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)


        preprocessing_obj = self.get_data_tranformation()

        target_col_name = 'price'
        drop_columns = [target_col_name]

        # Train data
        input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
        target_feature_train_df = train_df[target_col_name]
        
        #Test data
        input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
        target_feature_test_df = test_df[target_col_name]

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
        

        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

        return (train_arr,test_arr)

