from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.utils.common import load_object

app = FastAPI(title="ML Pipeline Production API")

artifact = load_object("artifacts/models/model.pkl")
model = artifact["model"]
preprocessor = artifact["preprocessor"]

class PredictionRequest(BaseModel):
    features: dict

@app.get("/")
def home():
    return {"message": "ML Pipeline API is running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.features])
    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]
    return {"prediction": int(prediction) if str(prediction).isdigit() else str(prediction)}
