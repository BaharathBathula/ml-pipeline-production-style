def test_pipeline_import():
    from src.pipelines.training_pipeline import TrainingPipeline
    assert TrainingPipeline is not None
