# predict v1.3:
#   English comments

import pickle
import pandas as pd


def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model


def get_prediction(model, cust_input, input_ids):
    input_df = pd.DataFrame(cust_input).transpose()
    input_df.columns = input_ids[3:]
    return model.predict(input_df)[0]


def load_model_and_predict(path, cust_input, input_ids):
    """
    Args:
        path: path and name of selected model file
        cust_input: values of selected model features
        input_ids: names of selected model features
    
    Returns:
        get_prediction: prediction built from Args.
    """
    model = load_model(path)
    return get_prediction(model, cust_input, input_ids)
