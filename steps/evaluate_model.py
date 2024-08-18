import logging
from typing import Annotated, Tuple

import mlflow
import numpy as np
import pandas as pd
# from src.modelEvaluation import Evaluator
# from sklearn.base import RegressorMixin
from zenml import get_step_context, log_artifact_metadata, step
from sklearn.metrics import mean_squared_error, r2_score
from scipy.sparse import csr_matrix
from zenml.client import Client
experiment_tracker = Client().active_stack.experiment_tracker


# @step
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
        model: object,
        X_test: csr_matrix,
        y_test: pd.Series,
) -> Tuple[Annotated[float, "r2"], Annotated[float, "rmse"]]:

    try:
        # prediction = model.predict(X_test)
        # evaluator = Evaluator()
        mlflow.set_tag("developer", "akan")

        lr = model
        prediction = lr.predict(X_test)
        r2 = r2_score(y_test, prediction)
        mse = mean_squared_error(y_test, prediction)
        rmse = np.sqrt(mse)

        # Log to MLFlow
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Also add the metrics to the Model within the ZenML Model Control Plane
        artifact = get_step_context().model.get_artifact("sklearn_REG")

        log_artifact_metadata(
            metadata={
                "metrics": {
                    "r2_score": float(r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                }
            },
            artifact_name=artifact.name,
            artifact_version=artifact.version,
        )
        return mse, rmse
    except Exception as e:
        logging.error(e)
        raise e
