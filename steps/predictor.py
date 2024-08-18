import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(input_data)
    data.pop("columns")
    data.pop("index")
    categorical = ['PU_DO']  # 'PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    columns_for_df = [
        "pickup_location",
        "dropoff_location",
        "trip_distance"
    ]
    st_df = pd.DataFrame(data["data"], columns=columns_for_df)
    dv = DictVectorizer()
    st_df = dv.transform(st_df)
    df = st_df[categorical + numerical].to_dict(orient='records')

    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction
