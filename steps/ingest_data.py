import os
import logging
import pandas as pd
from zenml import step


class DataIngestor:
    def __init__(self, taxi_type: str, year: int, month: int):
        """
        Ingesting the data from source
        :param month:
        :param year:
        :param taxi_type:
        :return: pd Dataframe
        """
        self.taxi_type = taxi_type
        self.year = year
        self.month = month
        self.local_path = f"{taxi_type}_{year:04d}-{month:04d}.parquet"
        self.data_path = (f"https://d37ci6vzurychx.cloudfront.net/trip-data/"
                          f"{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet")

    def fetchStore_data(self) -> pd.DataFrame:
        logging.info(f"Fetching data from {self.data_path} ...")
        if not os.path.exists(f"./datasets/{self.local_path}"):
            # Create the 'datasets' directory if it doesn't exist
            os.makedirs("./datasets", exist_ok=True)
            try:
                taxi_data = pd.read_parquet(self.data_path)
                logging.info(f"Storing to local drive: {self.local_path}")
                taxi_data.to_parquet(f"./datasets/{self.local_path}")
            except Exception as e:
                raise e
        else:
            logging.info(f"Loading data from Local")
            taxi_data = pd.read_parquet(f"./datasets/{self.local_path}")

        return taxi_data

    def sample_data(self) -> pd.DataFrame:
        full_df = self.fetchStore_data()
        df = full_df.sample(n=100000)
        return df


@step
def ingest_data(taxi_type: str, year: int, month: int) -> pd.DataFrame:
    """
    Ingest Data and return a Dataframe with the whole dataset.
    Returns: df: pd.DataFrame
    """
    try:
        data_ingestor = DataIngestor(taxi_type, year, month)
        df = data_ingestor.sample_data()
        return df
    except Exception as e:
        logging.error(e)
        raise e
