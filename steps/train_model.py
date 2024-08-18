import logging
from typing import Annotated
import mlflow
import pandas as pd
from zenml import ArtifactConfig, step
from src.modelDevelopment import ModelTrainer
from sklearn.base import RegressorMixin
from zenml.client import Client
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
experiment_tracker = Client().active_stack.experiment_tracker


# @step()
@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
        X_train: csr_matrix,
        y_train: pd.Series,
    # do_fine_tuning: bool = True
) -> Annotated[
    object,
    ArtifactConfig(name="sklearn_REG",
                   is_model_artifact=True),
]:
    try:
        # model_training = ModelTrainer(X_train, y_train, X_test, y_test)

        mlflow.sklearn.autolog()
        lr = LinearRegression()
        lr_model = lr.fit(X_train, y_train)

        # mean_squared_error(y_val, y_pred, squared=False)
        return lr_model
    except Exception as e:
        logging.error(e)
        raise e
