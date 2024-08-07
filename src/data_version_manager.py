import os
import pandas as pd
import numpy as np
from dvc.repo import Repo as DvcRepo
from dvc.exceptions import DvcException

from config import PROJECT_ROOT


class DataVersionManager:
    def __init__(self, project_root: str, data_path: str):
        self.project_root = project_root
        self.data_path = data_path
        self.full_data_path = os.path.join(self.project_root, data_path)
        self.dvc_repo = DvcRepo(project_root)

    def init_dvc(self):
        try:
            if not os.path.exists(os.path.join(self.project_root, '.dvc')):
                self.dvc_repo = DvcRepo.init(self.project_root)
            print("DVC initialized")
        except DvcException as e:
            print(f"Error initializing DVC: {e}")

    def add_data_to_dvc(self):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.full_data_path), exist_ok=True)

            # Check if the file exists, if not create an empty DataFrame and save it
            if not os.path.exists(self.full_data_path):
                empty_df = pd.DataFrame(
                    columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                             'DiabetesPedigreeFunction', 'Age', 'Outcome'])
                empty_df.to_csv(self.full_data_path, index=False)
                print(f"Created empty dataset at {self.full_data_path}")

            # Check if there's an overlap with an existing stage output
            if os.path.exists(os.path.join(self.project_root, 'dvc.yaml')):
                with open(os.path.join(self.project_root, 'dvc.yaml'), 'r') as f:
                    dvc_yaml_content = f.read()
                    if self.data_path in dvc_yaml_content:
                        print(
                            f"Data path '{self.data_path}' overlaps with an existing stage output. Running the pipeline to resolve.")
                        self.dvc_repo.reproduce()
                        return

            # Add the file to DVC
            self.dvc_repo.add(self.full_data_path)

            # Commit the changes
            self.dvc_repo.commit(f"Add/Update data in DVC: {self.data_path}")
            print(f"Added {self.data_path} to DVC")
        except DvcException as e:
            print(f"Error adding data to DVC: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def create_data_version(self, version_name: str):
        try:
            self.dvc_repo.commit(f"Create data version: {version_name}")
            self.dvc_repo.tag(version_name)
            print(f"Created DVC tag for version: {version_name}")
        except DvcException as e:
            print(f"Error creating data version: {e}")

    def switch_data_version(self, version_name: str):
        try:
            self.dvc_repo.checkout(version_name, force=True)
            print(f"Switched to DVC version: {version_name}")
        except DvcException as e:
            print(f"Error switching data version: {e}")

    def generate_sample_data(self, n_samples: int = 500):
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                   'DiabetesPedigreeFunction', 'Age', 'Outcome']

        data = pd.DataFrame({
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'Glucose': np.random.randint(70, 200, n_samples),
            'BloodPressure': np.random.randint(40, 120, n_samples),
            'SkinThickness': np.random.randint(0, 60, n_samples),
            'Insulin': np.random.randint(0, 500, n_samples),
            'BMI': np.random.uniform(18, 50, n_samples).round(1),
            'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples).round(3),
            'Age': np.random.randint(21, 80, n_samples),
            'Outcome': np.random.randint(0, 2, n_samples)
        })

        return data

    def create_or_update_data(self, version_name: str, n_samples: int = 500):
        data = self.generate_sample_data(n_samples)
        os.makedirs(os.path.dirname(self.full_data_path), exist_ok=True)
        data.to_csv(self.full_data_path, index=False)
        self.add_data_to_dvc()
        self.create_data_version(version_name)
        print(f"Created/Updated data version: {version_name}")

    def append_new_data(self, new_data: pd.DataFrame):
        if os.path.exists(self.full_data_path):
            existing_data = pd.read_csv(self.full_data_path)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data
        os.makedirs(os.path.dirname(self.full_data_path), exist_ok=True)
        updated_data.to_csv(self.full_data_path, index=False)
        print(f"Appended {len(new_data)} new samples to the dataset.")

    def create_new_version(self, version_name: str, n_samples: int = 100):
        new_data = self.generate_sample_data(n_samples)
        self.append_new_data(new_data)
        self.add_data_to_dvc()
        self.create_data_version(version_name)
        print(f"Created new version {version_name} with {n_samples} additional samples.")