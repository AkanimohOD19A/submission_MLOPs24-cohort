import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from run_deployment import run_main
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer


def main():
    st.title("End to End Customer Satisfaction Pipeline with ZenML")

    high_level_image = Image.open("_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open(
        "_assets/training_and_deployment_pipeline_updated.png"
    )

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
    | Models        | Description   | 
    | ------------- | -     | 
    | Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
    | Payment Installments   | Number of installments chosen by the customer. |  
    | Payment Value |       Total amount paid by the customer. | 
    | Price |       Price of the product. |
    | Freight Value |    Freight value of the product.  | 
    | Product Name length |    Length of the product name. |
    | Product Description length |    Length of the product description. |
    | Product photos Quantity |    Number of product published photos |
    | Product weight measured in grams |    Weight of the product measured in grams. | 
    | Product length (CMs) |    Length of the product measured in centimeters. |
    | Product height (CMs) |    Height of the product measured in centimeters. |
    | Product width (CMs) |    Width of the product measured in centimeters. |
    """
    )
    pickup_location = st.sidebar.slider("PickUp Location")
    dropoff_location = st.sidebar.slider("DropOff Location")
    trip_distance = st.number_input("Distance")

    if st.button("Predict"):
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        service = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )[0]
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "pickup_location": [pickup_location],
                "dropoff_location": [dropoff_location],
                "trip_distance": [trip_distance],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your likely trip Duration given provided details is :-{}".format(
                pred
            )
        )
    if st.button("Results"):
        st.write(
            "We have experimented with just simple linear regression models. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["Linear Regression Model"],
                "MSE": [1.804],
                "RMSE": [1.343],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("_assets/feature_importance_gain.png")
        st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()
