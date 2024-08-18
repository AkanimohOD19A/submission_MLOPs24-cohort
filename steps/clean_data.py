import logging
from typing import Tuple, Any
import pandas as pd
from src.dataCleaning import DataCleaning
from typing_extensions import Annotated
from zenml import step


@step
def clean_data(data: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
    try:
        data_cleaning = DataCleaning(data)
        df = data_cleaning.preprocess_data()
        X_train, X_test, y_train, y_test = data_cleaning.split_data(df)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e
