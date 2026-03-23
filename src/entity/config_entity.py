from dataclasses import dataclass


@dataclass
class TrainingPipelineConfig:
    project_name: str
    raw_data_path: str
    train_path: str
    test_path: str
    target_column: str
    test_size: float
    random_state: int
    model_path: str
    metrics_path: str
    experiment_name: str
