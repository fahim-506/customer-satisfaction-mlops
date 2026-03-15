import logging
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
        logging.info("Reading dataset from CSV file...")
        df = pd.read_csv(self.data_path)

        if df.empty:
            raise ValueError("Loaded dataset is empty")

        logging.info(f"Dataset loaded successfully with shape {df.shape}")
        return df


@step
def ingest_data() -> pd.DataFrame:
    """
    ZenML step to ingest data.

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        ingestor = IngestData("./data/olist_customers_dataset.csv")
        df = ingestor.get_data()
        return df

    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise