import pandas as pd
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class StandardScalingStrategy(FeatureEngineeringStrategy):
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
