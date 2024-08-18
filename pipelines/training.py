from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.promote_model import promote_model
from zenml import pipeline


@pipeline
def training_pipeline(taxi_type: str, year: int, month: int):
    """Training Pipeline.

    Args:
        taxi_type: Type of taxi data to use.
        year: Year of the data.
        month: Month of the data.
        config: Configuration dictionary.
    """
    df = ingest_data(taxi_type, year, month)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
    )

    mse, rmse = evaluate_model(model, X_test, y_test)
    # is_promoted = promote_model(mse=mse)
    return model#, is_promoted
