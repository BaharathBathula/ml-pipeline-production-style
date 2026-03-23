import pandas as pd
from src.utils.logger import logger


class DataValidation:
    def __init__(self, required_columns=None):
        self.required_columns = required_columns or []

    def validate(self, file_path: str):
        logger.info("Starting data validation")
        df = pd.read_csv(file_path)

        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            raise ValueError("Input dataframe is empty")

        logger.info("Data validation completed successfully")
        return True
