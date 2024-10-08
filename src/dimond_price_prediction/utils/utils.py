from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.dimond_price_prediction.exception.exception import CustomException
import sys
import os
import pickle

def evaluate_model(X_train,y_train,X_test,y_test,models):
    reports = {}
    for i in range(len(models)):
        model = list(models.values())[i]
        model.fit(X_train,y_train)

        y_test_pred = model.predict(X_test)

        test_model_score = r2_score(y_test,y_test_pred)

        reports[list(models.keys())[i]] = test_model_score

    return reports

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e, sys)    


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)