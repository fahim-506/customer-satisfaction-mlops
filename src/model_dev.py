from utils.logger import logger
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self,X_train,y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

class LinearRegressionModel(Model):
    """
    Linear Reggression Model
    """
    def train(self,X_train,y_train,**kwargs):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        try:
            # Impute numeric NaNs (redundant if already cleaned)
            imputer = SimpleImputer(strategy="median")
            X_train_imputed = imputer.fit_transform(X_train)

            reg = LinearRegression(**kwargs)
            reg.fit(X_train_imputed, y_train)
            logger.info("Model Training Completed")
            return reg
        except Exception as e:
            logger.error(f"Error in training model: {e}")
            raise e