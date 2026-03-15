from utils.logger import logger
import pandas as pd
from zenml import step

@step
def evaluate_model(df:pd.DataFrame) -> None:
    pass