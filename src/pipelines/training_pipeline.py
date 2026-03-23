import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.common import read_yaml
from src.utils.logger import logger

class TrainingPipeline:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = read_yaml(config_path)

    def run_pipeline(self):
        logger.info("Training pipeline started")

        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        train_df = pd.read_csv(train_path)
        target_col = self.config["data"]["target_column"]

        transformation = DataTransformation()
        preprocessor = transformation.get_transformer(train_df, target_col)

        trainer = ModelTrainer()
        metrics = trainer.initiate_model_trainer(train_path, test_path, preprocessor)

        logger.info(f"Training pipeline completed. Metrics: {metrics}")
        print("Training completed successfully")
        print(metrics)
