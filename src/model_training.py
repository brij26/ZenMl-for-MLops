from zenml import step
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Annotated


@step
def train_model(X_train: np.ndarray, y_train: pd.Series) -> Annotated[RandomForestClassifier, 'Randomforest model']:
    """Train the model"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
