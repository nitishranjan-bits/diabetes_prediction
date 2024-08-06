import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
import pandas as pd
from typing import Dict, Any, Optional


class ModelRegistry:
    def __init__(self):
        self.client = MlflowClient()

    def register_model(self, model: Any, model_name: str, run_id: str, input_example: pd.DataFrame,
                       description: str = None, tags: Dict[str, str] = None) -> Optional[str]:
        try:
            # Infer the model signature
            signature = infer_signature(input_example, model.predict(input_example))

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            # Register the model
            model_uri = f"runs:/{run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            version = result.version

            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=version,
                    description=description
                )

            if tags:
                for key, value in tags.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=value
                    )

            print(f"Registered model: {model_name}, version: {version}")
            return version
        except Exception as e:
            print(f"Error registering model: {e}")
            return None

    # def get_model(self, model_name: str, model_version: str = 'latest') -> Any:
    #     if model_version == 'latest':
    #         versions = self.client.get_latest_versions(model_name, stages=["None"])
    #         if not versions:
    #             raise ValueError(f"No versions found for model {model_name}")
    #         model_version = versions[0].version
    #
    #     model_uri = f"models:/{model_name}/{model_version}"
    #     return mlflow.sklearn.load_model(model_uri)

    def get_model(self, model_name: str, model_version: str = 'latest') -> Any:
        if model_version == 'latest':
            query = f"name='{model_name}'"
            versions = self.client.search_model_versions(query)
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")

            # Sort versions by creation date or version number
            versions_sorted = sorted(versions, key=lambda x: x.version, reverse=True)
            model_version = versions_sorted[0].version

        model_uri = f"models:/{model_name}/{model_version}"
        return mlflow.sklearn.load_model(model_uri)

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Transitioned {model_name} version {version} to {stage}")

    def get_model_info(self, model_name: str, version: str = 'latest') -> Any:
        if version == 'latest':
            versions = self.client.get_latest_versions(model_name, stages=["None"])
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            version = versions[0].version

        return self.client.get_model_version(model_name, version)

    def update_model_description(self, model_name: str, version: str, description: str) -> None:
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        print(f"Updated description for {model_name} version {version}")

    def add_model_tag(self, model_name: str, version: str, key: str, value: str) -> None:
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=value
        )
        print(f"Added tag {key}={value} to {model_name} version {version}")

    def get_model_schema(self, model_name: str, version: str = 'latest') -> Dict[str, Any]:
        model_version = self.get_model_info(model_name, version)
        return model_version.signature.to_dict()
