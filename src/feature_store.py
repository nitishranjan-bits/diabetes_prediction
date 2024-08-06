import pandas as pd
from typing import Dict, Any


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
