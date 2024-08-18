import logging
import pandas as pd
from typing import Union, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import scipy.sparse

class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes the DataCleaning class."""
        self.data = data

    def preprocess_data(self) -> pd.DataFrame:
        """
        Strategy for preprocessing data
        """
        try:
            categorical = ['PULocationID', 'DOLocationID']

            self.data['tpep_dropoff_datetime'] = pd.to_datetime(self.data['tpep_dropoff_datetime'])
            self.data['tpep_pickup_datetime'] = pd.to_datetime(self.data['tpep_pickup_datetime'])

            self.data['duration'] = self.data['tpep_dropoff_datetime'] - self.data['tpep_pickup_datetime']
            self.data.duration = self.data.duration.apply(lambda td: td.total_seconds() / 60)
            self.data = self.data[(self.data.duration >= 1) & (self.data.duration <= 60)]
            self.data[categorical] = self.data[categorical].astype(str)

            self.data['PU_DO'] = self.data['PULocationID'] + '_' + self.data['DOLocationID']

            return self.data
        except Exception as e:
            logging.error(e)
            raise e

    def split_data(self, data: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        try:
            cat = ['PU_DO']
            num = ['trip_distance']

            dv = DictVectorizer()

            X = data.drop('duration', axis=1)
            y = data['duration']
            x_train, x_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )

            train_dicts = x_train[cat + num].to_dict(orient='records')
            X_train = dv.fit_transform(train_dicts)  # fit_transform
            print(type(X_train))
            test_dicts = x_test[cat + num].to_dict(orient='records')
            X_test = dv.transform(test_dicts)  # transform
            print(type(X_test))

            # Convert sparse matrices to DataFrames
            # X_train_df = pd.DataFrame.sparse.from_spmatrix(X_train)
            # X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e
