# mlops_pipeline.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import optuna
import dvc.api
from abc import ABC, abstractmethod
import subprocess
import os
import json
import datetime
from typing import Dict, Any, List

# Abstract Factory Pattern for ML Models
class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        pass

class RandomForestFactory(ModelFactory):
    def create_model(self, params: Dict[str, Any]):
        return RandomForestClassifier(**params)

class LogisticRegressionFactory(ModelFactory):
    def create_model(self, params: Dict[str, Any]):
        return LogisticRegression(**params)

class SVMFactory(ModelFactory):
    def create_model(self, params: Dict[str, Any]):
        return SVC(**params)

# Strategy Pattern for Feature Engineering
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class StandardScalingStrategy(FeatureEngineeringStrategy):
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Data Version Control Manager
class DataVersionManager:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def init_dvc(self):
        subprocess.run(["dvc", "init"], check=True)
        print("DVC initialized")

    def add_data_to_dvc(self):
        subprocess.run(["dvc", "add", self.data_path], check=True)
        subprocess.run(["git", "add", f"{self.data_path}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", "Add data to DVC"], check=True)
        print(f"Added {self.data_path} to DVC")

    def create_data_version(self, version_name: str):
        subprocess.run(["dvc", "tag", "add", "-d", self.data_path, version_name], check=True)
        subprocess.run(["git", "add", ".dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Create DVC version: {version_name}"], check=True)
        print(f"Created DVC version: {version_name}")

    def switch_data_version(self, version_name: str):
        subprocess.run(["dvc", "checkout", version_name], check=True)
        print(f"Switched to DVC version: {version_name}")

#
# class StandardScalingStrategy(FeatureEngineeringStrategy):
#     def engineer_features(self, data):
#         scaler = StandardScaler()
#         return scaler.fit_transform(data)

# Feature Store
class FeatureStore:
    def __init__(self):
        self.offline_store = {}
        self.online_store = {}

    def add_offline_feature(self, name: str, data: pd.DataFrame):
        self.offline_store[name] = data

    def get_offline_feature(self, name: str) -> pd.DataFrame:
        return self.offline_store.get(name)

    def add_online_feature(self, name: str, data: Dict[str, Any]):
        self.online_store[name] = data

    def get_online_feature(self, name: str) -> Dict[str, Any]:
        return self.online_store.get(name)

# Singleton Pattern for FeatureStore
# class FeatureStore:
#     _instance = None
#
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             cls._instance.features = {}
#         return cls._instance
#
#     def add_feature(self, name, data):
#         self.features[name] = data
#
#     def get_feature(self, name):
#         return self.features.get(name)
#

# Model Registry
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

# MLOps Pipeline
class MLOpsPipeline:
    def __init__(self, data_path: str, model_factory: ModelFactory, feature_engineering_strategy: FeatureEngineeringStrategy):
        self.data_path = data_path
        self.model_factory = model_factory
        self.feature_engineering_strategy = feature_engineering_strategy
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.data_version_manager = DataVersionManager(data_path)
    def load_data(self) -> pd.DataFrame:
        with dvc.api.open(self.data_path, mode='r') as f:
            df = pd.read_csv(f)
        return df

    def preprocess_data(self, df: pd.DataFrame):
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_engineered = self.feature_engineering_strategy.engineer_features(X)
        self.feature_store.add_offline_feature('engineered_features', X_engineered)
        return train_test_split(X_engineered, y, test_size=0.2, random_state=42)

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                                 y_test: pd.Series):
        def objective(trial):
            if isinstance(self.model_factory, RandomForestFactory):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                    'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif isinstance(self.model_factory, LogisticRegressionFactory):
                params = {
                    'C': trial.suggest_loguniform('C', 1e-5, 1e5),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
                }
            elif isinstance(self.model_factory, SVMFactory):
                params = {
                    'C': trial.suggest_loguniform('C', 1e-5, 1e5),
                    'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                    'gamma': trial.suggest_loguniform('gamma', 1e-5, 1e5)
                }

            model = self.model_factory.create_model(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)

        tuning_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        with open('hyperparameter_tuning_results.json', 'w') as f:
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
        if data_version:
            self.data_version_manager.switch_data_version(data_version)

        mlflow.set_experiment("MLOps Pipeline")

        with mlflow.start_run():
            mlflow.log_param("data_version", data_version if data_version else "latest")

            df = self.load_data()
            X_train, X_test, y_train, y_test = self.preprocess_data(df)

            best_params = self.optimize_hyperparameters(X_train, y_train, X_test, y_test)
            mlflow.log_params(best_params)

            model = self.train_model(X_train, y_train, best_params)

            metrics = self.evaluate_model(model, X_test, y_test)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            model_path = "models/model_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mlflow.sklearn.save_model(model, model_path)

            model_name = type(model).__name__
            model_version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.model_registry.register_model(model_name, model_version, model_path)

            print(f"Model training completed. Metrics: {metrics}")
            print(f"Model saved and registered: {model_name}, version: {model_version}")

            # Add some online features
            for i in range(10):
                self.feature_store.add_online_feature(f'online_feature_{i}', {
                    'value': np.random.rand(),
                    'timestamp': datetime.datetime.now().isoformat()
                })

    def predict(self, features: List[float], model_name: str, model_version: str = 'latest'):
        model = self.model_registry.get_model(model_name, model_version)

        # Fetch online features
        online_features = [self.feature_store.get_online_feature(f'online_feature_{i}')['value'] for i in range(10)]

        # Combine input features with online features
        all_features = np.array(features + online_features).reshape(1, -1)

        return model.predict(all_features)[0]

# Usage
if __name__ == "__main__":
    data_path = "../data/diabetes.csv"

    # Initialize DVC and add initial data
    data_version_manager = DataVersionManager(data_path)
    data_version_manager.init_dvc()
    data_version_manager.add_data_to_dvc()
    data_version_manager.create_data_version("v1")

    # Run pipeline with Random Forest
    rf_pipeline = MLOpsPipeline(data_path, RandomForestFactory(), StandardScalingStrategy())
    rf_pipeline.run(data_version="v1")

    # Simulate data update
    print("Simulating data update...")
    os.utime(data_path, None)
    data_version_manager.add_data_to_dvc()
    data_version_manager.create_data_version("v2")

    # Run pipeline with Logistic Regression on updated data
    lr_pipeline = MLOpsPipeline(data_path, LogisticRegressionFactory(), StandardScalingStrategy())
    lr_pipeline.run(data_version="v2")

    # Make a prediction using the latest RandomForest model
    sample_features = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    prediction = rf_pipeline.predict(sample_features, "RandomForestClassifier")
    print(f"Prediction for sample features: {prediction}")

    print("MLOps pipeline execution completed.")
