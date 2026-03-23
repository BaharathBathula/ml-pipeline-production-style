import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.utils.common import read_yaml

class DataTransformation:
    def __init__(self, config_path="configs/config.yaml"):
        self.config = read_yaml(config_path)

    def get_transformer(self, df, target_column):
        numerical_cols = df.drop(columns=[target_column]).select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.drop(columns=[target_column]).select_dtypes(include=["object"]).columns

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        return preprocessor
