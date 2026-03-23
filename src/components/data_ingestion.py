import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.common import read_yaml
from src.utils.logger import logger

class DataIngestion:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = read_yaml(config_path)

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion")
        df = pd.read_csv(self.config["data"]["raw_data_path"])

        train_df, test_df = train_test_split(
            df,
            test_size=self.config["model"]["test_size"],
            random_state=self.config["model"]["random_state"]
        )

        train_path = self.config["data"]["train_path"]
        test_path = self.config["data"]["test_path"]

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Data ingestion completed")
        return train_path, test_path
