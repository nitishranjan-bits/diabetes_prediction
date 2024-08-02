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
from data_version_manager import DataVersionManager
from feature_engineering import StandardScalingStrategy, FeatureEngineeringStrategy
from model_factories import RandomForestFactory, LogisticRegressionFactory, SVMFactory, ModelFactory
from feature_store import FeatureStore
from model_registry import ModelRegistry

try:
    from optuna.integration import SklearnPruningCallback
    pruning_callback_available = True
except ImportError:
    print("Warning: SklearnPruningCallback is not available. Pruning will be disabled.")
    pruning_callback_available = False

# MLOps Pipeline
class MLOpsPipeline:
    def __init__(self, data_path: str, model_factory: ModelFactory, feature_engineering_strategy: FeatureEngineeringStrategy):
        self.data_path = data_path
        self.model_factory = model_factory
        self.feature_engineering_strategy = feature_engineering_strategy
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.data_version_manager = DataVersionManager(data_path)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.n_features = None

    def load_data(self) -> pd.DataFrame:
        data_path = os.path.join(self.project_root, self.data_path)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return pd.read_csv(data_path)
        # data_path = self.project_root + "/data/diabetes.csv"
        # df = pd.read_csv(data_path)
        # return df

    # def load_data(self) -> pd.DataFrame:
    #     try:
    #         # Attempt to open the file with DVC
    #         with dvc.api.open(self.data_path, mode='r') as f:
    #             df = pd.read_csv(f)
    #         return df
    #     except dvc.exceptions.PathMissingError:
    #         print(f"Error: The path '{self.data_path}' does not exist in the DVC repository.")
    #         raise
    #     except dvc.exceptions.FileMissingError as e:
    #         print(f"Error: {str(e)}")
    #         raise
    #     except Exception as e:
    #         print(f"Unexpected error: {str(e)}")
    #         raise

    def preprocess_data(self, df: pd.DataFrame):
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_engineered = self.feature_engineering_strategy.engineer_features(X)
        self.feature_store.add_offline_feature('engineered_features', X_engineered)
        self.n_features = X_engineered.shape[1]
        return train_test_split(X_engineered, y, test_size=0.2, random_state=42)

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                                 y_test: pd.Series):
        def objective(trial):
            try:
                if isinstance(self.model_factory, RandomForestFactory):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                    }
                elif isinstance(self.model_factory, LogisticRegressionFactory):
                    params = {
                        'C': trial.suggest_float('C', 1e-5, 1e5),
                        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                        'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
                    }
                elif isinstance(self.model_factory, SVMFactory):
                    params = {
                        'C': trial.suggest_float('C', 1e-5, 1e3, log=True),  # Limited range
                        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
                        'gamma': trial.suggest_float('gamma', 1e-5, 1e3, log=True)  # Limited range
                    }

                model = self.model_factory.create_model(params)

                if pruning_callback_available:
                    pruning_callback = SklearnPruningCallback(trial, "accuracy")
                    model.fit(X_train, y_train, callbacks=[pruning_callback])
                else:
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                return accuracy_score(y_test, y_pred)
            except Exception as e:
                print(f"Trial failed with params: {trial.params}, error: {str(e)}")
                return 0.0  # Return a low score to indicate failure

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, timeout=600)

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
        if data_version:
            self.data_version_manager.switch_data_version(data_version)

            # Pull latest changes before running the pipeline
        # self.data_version_manager.pull_from_remote()

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
        # self.data_version_manager.push_to_remote()

    def predict(self, features: List[float], model_name: str, model_version: str = 'latest'):
        model = self.model_registry.get_model(model_name, model_version)

        # Fetch online features
        online_features = [self.feature_store.get_online_feature(f'online_feature_{i}')['value'] for i in range(10)]

        # Combine input features with online features
        # all_features = np.array(features + online_features).reshape(1, -1)
        all_features = np.array(features + online_features[:self.n_features - len(features)]).reshape(1, -1)


        return model.predict(all_features)[0]

# Usage
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = "data/diabetes.csv"

    # Initialize DVC and add initial data
    # data_version_manager = DataVersionManager(data_path)
    # data_version_manager.init_dvc()
    # data_version_manager.add_data_to_dvc()
    # data_version_manager.create_data_version("v1")

    # Run pipeline with Random Forest
    rf_pipeline = MLOpsPipeline(data_path, RandomForestFactory(), StandardScalingStrategy())
    rf_pipeline.run(data_version="v1")

    # Simulate data update

    print("Simulating data update...")
    full_data_path = os.path.join(project_root, data_path)
    if os.path.exists(full_data_path):
        os.utime(full_data_path, None)
        print(f"Updated timestamp of {full_data_path}")
    else:
        print(f"File not found: {full_data_path}")
    # data_version_manager.add_data_to_dvc()
    # data_version_manager.create_data_version("v2")

    # Run pipeline with Logistic Regression on updated data
    lr_pipeline = MLOpsPipeline(data_path, LogisticRegressionFactory(), StandardScalingStrategy())
    lr_pipeline.run(data_version="v2")

    # Run pipeline with Logistic Regression on updated data
    lr_pipeline = MLOpsPipeline(data_path, SVMFactory(), StandardScalingStrategy())
    lr_pipeline.run(data_version="v2")

    # Make a prediction using the latest RandomForest model

    features = [0.5, 0.2, 0.1, 0.7, 0.3, 0.6, 0.8, 0.4]  # Example input features
    prediction = lr_pipeline.predict(features, model_name='LogisticRegression', model_version='latest')

    print(f"Prediction for sample features: {prediction}")

    print("MLOps pipeline execution completed.")
