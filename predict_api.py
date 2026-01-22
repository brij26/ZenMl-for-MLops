from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from src.model_loader import load_latest_model_and_scaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend origin later (e.g. http://localhost:3000)
    allow_credentials=True,
    allow_methods=["*"],  # IMPORTANT: allows OPTIONS
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {"status": "API running"}


# Load model ONCE at startup (important)
model, scaler = load_latest_model_and_scaler()


@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array([
        req.sepal_length,
        req.sepal_width,
        req.petal_length,
        req.petal_width
    ]).reshape(1, -1)

    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    return {"prediction": int(prediction[0])}
