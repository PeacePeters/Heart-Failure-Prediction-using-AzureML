import json
import os
import pickle
import pandas as pd
import numpy as np
from azureml.core import Model
from sklearn.externals import joblib

test_data_sample=pd.DataFrame(data=[{"Age":75, "anaemia":0,"creatinine_phosphokinase":582,"diabetes":0, "ejection_fraction":20, "high_blood_pressure":1, "platelets":265000, "serum_creatinine":1.9, "serum_sodium":130, "sex":1, "smoking":0}])

input_sample = pd.DataFrame({"Age": pd.Series([0], dtype="float64"), "anaemia": pd.Series([0], dtype="int64"), "creatinine_phosphokinase": pd.Series([0], dtype="int64"), "diabetes": pd.Series([0], dtype="int64"), "ejection_fraction": pd.Series([0], dtype="int64"), "high_blood_pressure": pd.Series([0], dtype="int64"), "platelets": pd.Series([0], dtype="float64"), "serum_sodium": pd.Series([0], dtype="int64"), "sex": pd.Series([0], dtype="int64"), "smoking": pd.Series([0], dtype="int64")})
output_sample = np.array([0])


def init():
    """Function to load the model into a global object"""
    try:   
        global model
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'fitted_model.pkl')
        model = joblib.load(model_path)
        #model_name = 'heart_failure_automl_model'
        #model_path = Model.get_model_path(model_name=model_name)
        #model = joblib.load(model_path)

    except Exception as e:
        error = str(e)
        

def run(data):
    """Function to use model to predict a value based on the input data
        
        Args:
            None
        
        Returns:
            list: predicted values
            
        """
    try:
        inputs = json.loads(data)
        # make prediction
        result = model.predict(pd.DataFrame(inputs['data']))
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
