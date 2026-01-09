from zenml import step
from typing import Annotated
import pandas as pd


@step
def load_data() -> Annotated[pd.DataFrame, 'Raw-data']:
    """Load the dataset"""
    # For this example we will use iris dataset
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df
