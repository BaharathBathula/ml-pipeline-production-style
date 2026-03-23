import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.common import read_yaml, save_object, save_json
from src.utils.logger import logger


class ModelTrainer:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = read_yaml(config_path)

    def _prepare_target(self, y):
        if y.dtype == "object":
            return y.map({"Yes": 1, "No": 0}).fillna(y)
        return y

    def initiate_model_trainer(self, train_path, test_path, preprocessor):
        logger.info("Starting model training")

        target_col = self.config["data"]["target_column"]

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop(columns=[target_col])
        y_train = self._prepare_target(train_df[target_col])

        X_test = test_df.drop(columns=[target_col])
        y_test = self._prepare_target(test_df[target_col])

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=self.config["model"]["random_state"])),
            ]
        )

        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run():
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_proba)),
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")

            save_object(self.config["model"]["model_path"], pipeline)
            save_json(self.config["metrics"]["metrics_path"], metrics)

        logger.info(f"Model training completed with metrics: {metrics}")
        return metrics
