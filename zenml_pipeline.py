from zenml import pipeline
from src.data_ingestion import load_data
from src.feature_engineering import preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model


@pipeline
def training_pipeline():
    """Define the pipeline step"""
    df = load_data()
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    training_pipeline()
