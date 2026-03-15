from utils.logger import logger
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract Class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculate the scores for the model 

        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Return:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE)
    """
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logger.info('Calculating MSE')
            mse = mean_squared_error(y_true,y_pred)
            logger.info('MSE: {}'.format(mse))
            return mse
        except Exception as e:
            logger.error("Error in calculating MSE: {}".format(e))
            raise e
            
class R2(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logger.info('Calculating R2 Score')
            r2 = r2_score(y_true,y_pred) 
            logger.info("R2 score: {}".format(r2))
            return r2
        except Exception as e:
            logger.error('Error in calculating R2 score: {}'.format(e))
            raise e     

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Square Error
    """
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        try:
            logger.info('Calculating MSE')
            rmse = root_mean_squared_error(y_true,y_pred)
            logger.info('RMSE: {}'.format(rmse))
            return rmse
        except Exception as e:
            logger.error("Error in calculating RMSE: {}".format(e))
            raise e