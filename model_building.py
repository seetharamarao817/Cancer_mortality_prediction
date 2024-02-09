import numpy as np
import pickle
import os
from data_ingestion import DataIngestion
from handling_outliers import find_outliers
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
from hyperparameter_tunning import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin


def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series, Hyperparameter=True,model_name="linear_regression") -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None
        y_train = y_train.to_numpy().reshape(-1, 1)


        if model_name == "lightgbm":
            model = LightGBMModel()
        elif model_name == "randomforest":
            model = RandomForestModel()
        elif model_name == "xgboost":
            model = XGBoostModel()
        elif model_name == "linear_regression":
            model = LinearRegressionModel()
        else:
            print("Model name not supported")
        
        if Hyperparameter == True:
            print("starting hyperparameter tuning ")
            tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
            directory = f'/workspaces/Cancer_mortality_prediction/artifacts/{model_name}/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(os.path.join(directory, 'model.pkl'), 'wb') as files:
                pickle.dump(trained_model, files)
            return trained_model
        
        else:
            trained_model = model.train(x_train,y_train)
            directory = f'/workspaces/Cancer_mortality_prediction/artifacts/{model_name}/'
            if not os.path.exists(directory):
                os.makedirs(directory)

            with open(os.path.join(directory, 'model.pkl'), 'wb') as files:
                pickle.dump(trained_model, files)
            return trained_model
        
    except Exception as e:
        raise e
