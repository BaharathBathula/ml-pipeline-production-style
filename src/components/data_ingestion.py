import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.common import read_yaml, ensure_directory
from src.utils.logger import logger


class DataIngestion:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = read_yaml(config_path)

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion")

        raw_data_path = self.config["data"]["raw_data_path"]
        train_path = self.config["data"]["train_path"]
        test_path = self.config["data"]["test_path"]

        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw dataset not found at path: {raw_data_path}")

        df = pd.read_csv(raw_data_path)

        ensure_directory(os.path.dirname(train_path))
        ensure_directory(os.path.dirname(test_path))

        train_df, test_df = train_test_split(
            df,
            test_size=self.config["model"]["test_size"],
            random_state=self.config["model"]["random_state"],
            stratify=df[self.config["data"]["target_column"]] if self.config["data"]["target_column"] in df.columns else None,
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Data ingestion completed")
        return train_path, test_path
