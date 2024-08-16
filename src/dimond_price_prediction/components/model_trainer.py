"""This file will train the models."""
from webbrowser import get
from dimond_price_prediction.logger.logging import logging
from dimond_price_prediction.exception.exception import CustomException

import os 
from dataclasses import dataclass
from src.dimond_price_prediction.utils.utils import save_object
from src.dimond_price_prediction.components.model_evaluation import Evaluator

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


@dataclass
class TrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class Trainer:
    def __init__(self):
        self.model_trainer_config = TrainerConfig()

    def initate_data_training(self, train_array, test_array):
        X_train, y_train, X_test, y_test = (
            train_array[:,:-1],
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

        scores = {}
        evaluator = Evaluator()
        for i in range(list(models)):
            model_name, model = list(models.items())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            scores[model_name] = evaluator.eval_metrics(y_test,y_pred)
       
        best_model_name= max(scores,key=get)

        best_model = models[best_model_name]

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model
        )
        
        logging.info("Trained model saved.")