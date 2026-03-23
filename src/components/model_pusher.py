from src.utils.logger import logger


class ModelPusher:
    def __init__(self):
        pass

    def push_model(self, model_path: str):
        logger.info(f"Model ready for deployment: {model_path}")
        return model_path
