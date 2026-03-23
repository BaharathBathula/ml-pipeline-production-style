import pandas as pd
from sklearn.metrics import classification_report

from src.utils.logger import logger


class ModelEvaluation:
    def __init__(self, pipeline, target_column):
        self.pipeline = pipeline
        self.target_column = target_column

    def evaluate(self, test_path):
        logger.info("Starting model evaluation")

        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column].map({"Yes": 1, "No": 0}).fillna(test_df[self.target_column])

        predictions = self.pipeline.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)

        logger.info("Model evaluation completed")
        return report
