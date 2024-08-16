import os
import sys
import pandas as pd
from src.dimond_price_prediction.exception.exception import CustomException
from src.dimond_price_prediction.logger import logging
from src.dimond_price_prediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
        model_path=os.path.join("artifacts","model.pkl")
            
        preprocessor=load_object(preprocessor_path)
        model=load_object(model_path)
            
        scaled_data=preprocessor.transform(features)
            
        pred=model.predict(scaled_data)
            
        return pred
            
    
    
    
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
            
                
    def get_data_as_dataframe(self):
        custom_data_input_dict = {
                    'carat':[self.carat],
                    'depth':[self.depth],
                    'table':[self.table],
                    'x':[self.x],
                    'y':[self.y],
                    'z':[self.z],
                    'cut':[self.cut],
                    'color':[self.color],
                    'clarity':[self.clarity]
                }
        df = pd.DataFrame(custom_data_input_dict)
        return df
         