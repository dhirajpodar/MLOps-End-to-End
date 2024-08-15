"""This file will train the models."""
from DimondPricePrediction.logger.logging import logging
from DimondPricePrediction.exception.exception import CustomException

import os 
from dataclasses import dataclass
from DimondPricePrediction.utils.utils import save_object
from DimondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class TrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class Trainer:
    def __init__(self):
        self.model_trainer_config = TrainerConfig()

    def initate_data_training(self, train_array, test_array):
        X_train, y_train, X_test, y_test = (
            train_array[:,:,-1],
            train_array[:,-1],
            test_array[:,:,-1],
            test_array[:,-1],
        )    

        models = {
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }

        model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
        print(model_report)
        print('\n==========')
        logging.info(f"Model report::{model_report}")

        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        
        logging.info("Trained model saved.")