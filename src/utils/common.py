import yaml
import joblib
import json
import os

def read_yaml(path: str):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def save_object(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_object(path: str):
    return joblib.load(path)

def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
