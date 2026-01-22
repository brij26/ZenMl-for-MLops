from zenml.client import Client
from zenml import step
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Annotated, Tuple


@step
def load_latest_model_and_scaler() -> Tuple[Annotated[RandomForestClassifier, "model"], Annotated[StandardScaler, "scaler"]]:
    client = Client()

    # 3. Load model
    artifact = Client().get_artifact_version(
        "1bd99e6e-b321-48b3-8955-31cf50a5b60d")
    model = artifact.load()

    # 4. Load scaler
    scaler_artifact = Client().get_artifact_version(
        "7f88dbdc-7d6a-444a-bdfa-896bae96b0e2")
    scaler = scaler_artifact.load()

    print("Model loaded:", model)
    print("Scaler loaded:", scaler)

    return model, scaler


# if __name__ == "__main__":
#    model, scaler = load_latest_model_and_scaler()
