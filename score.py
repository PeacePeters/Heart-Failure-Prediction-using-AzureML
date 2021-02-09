import json
import numpy as np
import os
import pickle
import pandas as pd
from azureml.core import Model
from sklearn.externals import joblib

test_data_sample=pd.DataFrame(data=[{"Age":75, "anaemia":0,"creatinine_phosphokinase":582,"diabetes":0, "ejection_fraction":20, "high_blood_pressure":1, "platelets":265000, "serum_creatinine":1.9, "serum_sodium":130, "sex":1, "smoking":0, "time":4}])

def init():
    try:   
        global model
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'fitted_model.pkl')
        model = joblib.load(model_path)

    except Exception as e:
        error = str(e)

def run(data):
    try:
        inputs = json.loads(data)
        # make prediction
        result = model.predict(pd.DataFrame(inputs['data']))
        return result.tolist()
    except Exception as e:
        result = str(e)
        return result
