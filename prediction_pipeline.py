from zenml import pipeline, save_artifact
from src.model_loader import load_latest_model_and_scaler
from typing import Annotated
import numpy as np
from src.prediction_helper_functions import predict_step, create_features, map_prediction_to_class

# @pipeline(enable_cache=True)
# def prediction_pipeline():
#    """Pipeline to predict flower class from user input"""

#    sepal_length = float(input("Enter sepal length: "))
#    sepal_width = float(input("Enter sepal width: "))
#    petal_length = float(input("Enter petal length: "))
#    petal_width = float(input("Enter petal width: "))

#    # Combine features into a single sample (2D array)
#    input_data = np.array([[
#        sepal_length,
#        sepal_width,
#        petal_length,
#        petal_width
#    ]])

#    model, scaler = load_latest_model_and_scaler()

#    # Scale the input
#    scaled_input = scaler.transform(input_data)

#    # Predict
#    output = model.predict(scaled_input)

#    class_names = ["setosa", "versicolor", "virginica"]

#    predicted_class = class_names[output[0]]
#    return Annotated("Predicted Flower:", predicted_class)


@pipeline(enable_cache=True)
def prediction_pipeline(sepal_length: float = 5.1,
                        sepal_width: float = 3.5,
                        petal_length: float = 1.4,
                        petal_width: float = 0.4,) -> Annotated[str, "Flower Type"]:
    features = create_features(
        sepal_length, sepal_width, petal_length, petal_width)

    model, scaler = load_latest_model_and_scaler()
    prediction = predict_step(model, scaler, features)
    flower_name = map_prediction_to_class(prediction)
    return flower_name


if __name__ == "__main__":
    output = prediction_pipeline()
    print(output)
