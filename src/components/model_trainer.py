import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.common import read_yaml, save_object, save_json

class ModelTrainer:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = read_yaml(config_path)

    def initiate_model_trainer(self, train_path, test_path, preprocessor):
        target_col = self.config["data"]["target_column"]

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        model = LogisticRegression(max_iter=1000)

        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

        with mlflow.start_run():
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0)
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            save_object(self.config["model"]["model_path"], {
                "model": model,
                "preprocessor": preprocessor
            })
            save_json(self.config["metrics"]["metrics_path"], metrics)

        return metrics
