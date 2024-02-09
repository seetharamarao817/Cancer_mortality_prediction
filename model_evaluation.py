from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score
from typing_extensions import Annotated

def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, rmse
    
