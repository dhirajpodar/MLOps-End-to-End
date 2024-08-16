from operator import index
from textwrap import indent
import pandas as pd
import numpy as np
from src.dimond_price_prediction.logger.logging import logging
from src.dimond_price_prediction.exception.exception import CustomException

import os 
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion started...')

        try:
            data = pd.read_csv(Path(os.path.join('data','diamonds.csv')))

            os.makedirs(os.path.dirname(os.path.join(
                self.ingestion_config.raw_data_path
            )),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data saved in artifact folder')

            
            train_data, test_data = train_test_split(data,test_size=0.25)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info('Data train, test splitted and saved')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("Exception during data ingestion")
            raise CustomException(e,sys)