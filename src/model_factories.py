from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Dict, Any


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
