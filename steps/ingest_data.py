from utils.logger import logger
import pandas as pd
from zenml import step


class IngestData:
    """
    Data ingestion class which loads data from a CSV file.
    """

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset
        """
        logger.info("Reading dataset from CSV file...")
        df = pd.read_csv(self.data_path)

        if df.empty:
            raise ValueError("Loaded dataset is empty")

        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df


@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    ZenML step to ingest data.

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        ingestor = IngestData(data_path)
        df = ingestor.get_data()
        return df

    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise