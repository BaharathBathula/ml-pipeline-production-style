from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

from src.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI(title="ML Pipeline Production API")

prediction_pipeline = None
try:
    prediction_pipeline = PredictionPipeline()
except Exception:
    prediction_pipeline = None


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/")
def home():
    return {"message": "ML Pipeline API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    global prediction_pipeline

    if prediction_pipeline is None:
        prediction_pipeline = PredictionPipeline()

    prediction = prediction_pipeline.predict(request.features)

    if str(prediction).isdigit():
        prediction = int(prediction)

    return {"prediction": prediction}
