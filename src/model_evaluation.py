import pandas as pd
import numpy as np
from zenml import step
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Annotated


@step
def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series) -> Annotated[float, 'Accuracy']:
    """Evaluate the model and return the accuracy"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy : {accuracy:.2f}")
    return accuracy
