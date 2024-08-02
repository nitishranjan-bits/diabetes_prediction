import mlflow.sklearn

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register_model(self, model_name: str, model_version: str, model_path: str):
        if model_name not in self.models:
            self.models[model_name] = {}
        self.models[model_name][model_version] = model_path
        print(f"Registered model: {model_name}, version: {model_version}")

    def get_model(self, model_name: str, model_version: str = 'latest'):
        if model_name in self.models:
            if model_version == 'latest':
                model_version = max(self.models[model_name].keys())
            return mlflow.sklearn.load_model(self.models[model_name][model_version])
        else:
            raise ValueError(f"Model {model_name} not found in registry")
