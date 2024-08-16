"""This file will train the models."""
import numpy as np
from src.dimond_price_prediction.logger.logging import logging
from src.dimond_price_prediction.exception.exception import CustomException

import os 
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.dimond_price_prediction.utils.utils import load_object

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

    def initate_model_evaluation(self, test_array):
        X_test, y_test = (test_array[:,:,-1], test_array[:,-1])

        model_path = os.path.join('artifacts','model.pkl')
        model=load_object(model_path)

        y_pred = model.predict(X_test)

        return y_test, y_pred