import pandas as pd
from src.utils.common import load_object


class PredictionPipeline:
    def __init__(self, model_path="artifacts/models/model.pkl"):
        self.pipeline = load_object(model_path)

    def predict(self, features: dict):
        input_df = pd.DataFrame([features])
        prediction = self.pipeline.predict(input_df)[0]
        return prediction
