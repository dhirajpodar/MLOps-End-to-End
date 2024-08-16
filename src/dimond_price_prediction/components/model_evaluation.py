"""This file will train the models."""

import os 
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dimond_price_prediction.utils.utils import load_object
import mlflow
import numpy as np
import dagshub

class Evaluator:
    def __init__(self):
        pass

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        print('Model Training Performance')
        print('RMSE:',rmse)
        print('MAE:',mae)
        print('R2:',r2)
        print('\n=======================')
        return rmse, mae, r2

    def save_model_evaluation_metrics(self, test_array):
        dagshub.init(repo_owner='dhirajpodar', repo_name='MLOps-End-to-End', mlflow=True)

        X_test, y_test = (test_array[:,:-1], test_array[:,-1])

        model_path = os.path.join('artifacts','model.pkl')
        model=load_object(model_path)
    
        with mlflow.start_run():

            y_pred = model.predict(X_test)
            (rmse, mae, r2) = self.eval_metrics(y_test, y_pred)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(model, "model")

