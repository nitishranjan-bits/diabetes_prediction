import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Dict, Any

class ModelFactory:
    def create_model(self, params: Dict[str, Any]):
        raise NotImplementedError

    def get_hyperparameter_space(self, trial):
        raise NotImplementedError

class RandomForestFactory(ModelFactory):
    def create_model(self, params: Dict[str, Any]) -> RandomForestClassifier:
        return RandomForestClassifier(**params)

    def get_hyperparameter_space(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }

class LogisticRegressionFactory(ModelFactory):
    def create_model(self, params: Dict[str, Any]) -> LogisticRegression:
        return LogisticRegression(**params)

    def get_hyperparameter_space(self, trial):
        return {
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }

class SVMFactory(ModelFactory):
    def create_model(self, params: Dict[str, Any]) -> SVC:
        params['max_iter'] = params.get('max_iter', 1000)
        params['tol'] = params.get('tol', 1e-3)
        params['cache_size'] = params.get('cache_size', 200)

        return SVC(**params)
    def get_hyperparameter_space(self, trial):
        # return {
        #     'C': trial.suggest_float('C', 1e-5, 1e3, log=True),
        #     'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
        #     'gamma': trial.suggest_float('gamma', 1e-5, 1e3, log=True)
        # }
        return {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'max_iter': trial.suggest_int('max_iter', 1000, 50000),
            'tol': trial.suggest_float('tol', 1e-4, 1e-2, log=True)
        }

    def preprocess_data(self, X, y):
        if X.shape[0] > 10000:
            indices = np.random.choice(X.shape[0], 10000, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]
        return X, y