import json
import os
from typing import Dict, Any

import mlflow
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from config import MLFLOW_TRACKING_URI, PROJECT_ROOT, MODEL_REGISTRY_URI, MLFLOW_EXPERIMENT_NAME
from data_version_manager import DataVersionManager
from feature_engineering import StandardScalingStrategy, FeatureEngineeringStrategy
from feature_store import FeatureStore
from model_factories import RandomForestFactory, LogisticRegressionFactory, SVMFactory, ModelFactory

try:
    from optuna.integration import SklearnPruningCallback

    pruning_callback_available = True
except ImportError:
    print("Warning: SklearnPruningCallback is not available. Pruning will be disabled.")
    pruning_callback_available = False

import logging

from model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MLOpsPipeline:
    def __init__(self, data_path: str,  model_factory: ModelFactory,
                 feature_engineering_strategy: FeatureEngineeringStrategy):
        self.data_path = data_path
        self.project_root = PROJECT_ROOT
        self.model_factory = model_factory
        self.feature_engineering_strategy = feature_engineering_strategy
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.data_version_manager = DataVersionManager(PROJECT_ROOT, data_path)
        self.n_features = None
        self.logger = logging.getLogger(self.__class__.__name__)

        # mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MODEL_REGISTRY_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        # self.mlflow_client = mlflow.tracking.MlflowClient()

    def load_data(self) -> pd.DataFrame:
        data_path = os.path.join(self.project_root, self.data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return pd.read_csv(data_path)

    def preprocess_data(self, df: pd.DataFrame):
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age']
        X = df[feature_names]
        y = df['Outcome']

        # Store important features in the feature store
        self.feature_store.add_offline_feature('raw_features', X)
        self.feature_store.add_offline_feature('target', y)

        X_engineered = self.feature_engineering_strategy.engineer_features(X)
        self.feature_store.add_offline_feature('engineered_features', X_engineered)

        return train_test_split(X_engineered, y, test_size=0.2, random_state=42)

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                                 y_test: pd.Series):
        def objective(trial):
            X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

            try:
                params = self.model_factory.get_hyperparameter_space(trial)
                model = self.model_factory.create_model(params)
                if pruning_callback_available:
                    pruning_callback = SklearnPruningCallback(trial, "accuracy")
                    model.fit(X_train_subset, y_train_subset, callbacks=[pruning_callback])
                else:
                    model.fit(X_train_subset, y_train_subset)

                return model.score(X_test, y_test)
            except Exception as e:
                print(f"Trial failed with params: {trial.params}, error: {str(e)}")
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=200, timeout=600)

        tuning_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        results_path = os.path.join(self.project_root, 'hyperparameter_tuning_results.json')
        with open(results_path, 'w') as f:
            json.dump(tuning_results, f, indent=2)

        return study.best_params

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]):
        model = self.model_factory.create_model(params)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }

    def run(self, data_version: str = None):
        try:
            # if data_version:
            #     try:
            #         self.data_version_manager.switch_data_version(data_version)
            #     except ValueError as e:
            #         self.logger.error(f"Data version error: {str(e)}")
            #         return

            # with mlflow.start_run(run_name=f"diabetes_{type(self.model_factory).__name__}") as run:
            with mlflow.start_run(run_name=f"diabetes_{type(self.model_factory).__name__}") as run:
                run_id = run.info.run_id

                mlflow.log_param("data_version", data_version if data_version else "latest")
                mlflow.log_param("model_type", type(self.model_factory).__name__)

                df = self.load_data()
                X_train, X_test, y_train, y_test = self.preprocess_data(df)

                if isinstance(self.model_factory, SVMFactory):
                    X_train, y_train = self.model_factory.preprocess_data(X_train, y_train)
                    X_test, y_test = self.model_factory.preprocess_data(X_test, y_test)

                best_params = self.optimize_hyperparameters(X_train, y_train, X_test, y_test)
                mlflow.log_params(best_params)

                model = self.train_model(X_train, y_train, best_params)

                metrics = self.evaluate_model(model, X_test, y_test)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # model_name = f"diabetes_{type(model).__name__.lower()}"
                model_name = f"diabetes_{type(self.model_factory).__name__}"
                description = f"Diabetes prediction model trained on data version {data_version} with best parameters: {best_params}"
                tags = {
                    "data_version": data_version if data_version else "latest",
                    "accuracy": f"{metrics['accuracy']:.4f}",
                    "f1_score": f"{metrics['f1']:.4f}",
                    "model_type": type(model).__name__,
                    "model_name": model_name,
                    "model_uri": f"runs:/{run_id}/{model_name}"
                }

                feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                                 'DiabetesPedigreeFunction', 'Age']
                try:
                    model_version = self.model_registry.register_model(
                        model,
                        model_name,
                        run_id,
                        X_train[feature_names].head(),
                        description=description,
                        tags=tags
                    )

                    if model_version:
                        if metrics['accuracy'] > 0.75 and metrics['f1'] > 0.60:
                            self.model_registry.transition_model_stage(model_name, model_version, "Staging")

                        print(f"Model training completed. Metrics: {metrics}")
                        print(f"Model saved and registered: {model_name}, version: {model_version}")
                    else:
                        print("Model registration failed. Please check the MLflow server configuration.")

                except Exception as e:
                    self.logger.error(f"Failed to register model: {str(e)}")
                    mlflow.log_param("model_registration_error", str(e))

                print(f"MLflow run ID: {run.info.run_id}")

        except Exception as e:
            self.logger.error(f"An error occurred during pipeline execution: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

    def predict(self, features: Dict[str, float], model_name: str, model_version: str = 'latest'):
        try:
            model = self.model_registry.get_model(model_name, model_version)
        except mlflow.exceptions.MlflowException as e:
            print(f"Error getting model: {e}")
            print("Make sure the model is registered and the MLflow server is running.")
            return None

        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age']
        feature_values = [float(features.get(name, 0)) for name in feature_names]
        input_data = pd.DataFrame([feature_values], columns=feature_names)
        return model.predict(input_data)[0]

#
# # Usage
# if __name__ == "__main__":
#     data_path = "data/diabetes.csv"
#
#     # Run pipeline with Random Forest
#     rf_pipeline = MLOpsPipeline(data_path, RandomForestFactory(), StandardScalingStrategy())
#     rf_pipeline.run(data_version="v1")
#
#     # Simulate data update
#     print("Simulating data update...")
#     if os.path.exists(data_path):
#         os.utime(data_path, None)
#
#     # Run pipeline with Logistic Regression on updated data
#     lr_pipeline = MLOpsPipeline(data_path, LogisticRegressionFactory(), StandardScalingStrategy())
#     lr_pipeline.run(data_version="v2")
#
#     # Run pipeline with SVM on updated data
#     svm_pipeline = MLOpsPipeline(data_path, SVMFactory(), StandardScalingStrategy())
#     svm_pipeline.run(data_version="v2")
#
#     features = {
#         'Pregnancies': 6.0,
#         'Glucose': 148.0,
#         'BloodPressure': 72.0,
#         'SkinThickness': 35.0,
#         'Insulin': 0.0,
#         'BMI': 33.6,
#         'DiabetesPedigreeFunction': 0.627,
#         'Age': 50.0
#     }
#     prediction = lr_pipeline.predict(features, model_name='diabetes_LogisticRegressionFactory', model_version='latest')
#     result = {'prediction': 'Diabetic' if prediction == 1 else 'Non-diabetic'}
#     print(f"Prediction for sample features: {result['prediction']}")
#
#     print("Diabetes prediction pipeline execution completed.")