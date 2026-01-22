from zenml import step
import numpy as np
from typing import Tuple, Annotated
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


@step
def predict_step(
    model: RandomForestClassifier,
    scaler: StandardScaler,
    features: np.ndarray
) -> int:
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return int(prediction[0])


@step
def create_features(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> Annotated[np.ndarray, "User Input"]:
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])


@step
def map_prediction_to_class(prediction: int) -> Annotated[str, "Class"]:
    class_map = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
    return class_map[prediction]
