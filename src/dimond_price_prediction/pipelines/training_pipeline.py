from src.dimond_price_prediction.components.data_ingestion import DataIngestion
from src.dimond_price_prediction.components.data_transformation import DataTransformation
from src.dimond_price_prediction.components.model_evaluation import Evaluator
from src.dimond_price_prediction.components.model_trainer import Trainer

import os
from src.dimond_price_prediction.logger.logging import logging
from src.dimond_price_prediction.exception.exception import CustomException

class TrainingPipeline:
    """Train pipeline."""

    def start_data_ingestion(self):
        logging.info("Data ingestion started...")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initate_data_ingestion()

        logging.info("Data ingestion done.")
        return train_data_path, test_data_path

    def start_data_transformation(self, train_data_path, test_data_path):
        logging.info("Data transformation started...")
        data_transformation = DataTransformation()

        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data ingestion done.")
        return train_arr, test_arr

    def start_model_training(self, train_arr, test_arr):
        logging.info("Config model training")
        trainer = Trainer()
        trainer.initate_data_training(train_arr, test_arr)


    def train(self):
        logging.info("Training started....")
        train_data_path, test_data_path = self.start_data_ingestion()
        train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
        self.start_model_training(train_arr,test_arr)