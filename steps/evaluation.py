from utils.logger import logger
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE,R2,RMSE
from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

logger.info(f"Tracker :: {experiment_tracker}")

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else "TRACKER 124")
def evaluate_model(model:RegressorMixin,
                   X_test:pd.DataFrame,
                   y_test:pd.Series) -> Tuple[
                       Annotated[float,'rmse'],
                       Annotated[float,'r2_score'],
                       ]:
    """
    Evaluates the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric('mse',mse)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)
        mlflow.log_metric('rmse',rmse)

        r2_class = R2()
        r2_score = r2_class.calculate_scores(y_test,prediction)
        mlflow.log_metric('r2_score',r2_score)

        return rmse,r2_score
    
    except Exception as e:
        logger.error(e)
        raise e