import os
import subprocess
import yaml
from typing import Optional

import pandas as pd
import numpy as np
from git import Repo as GitRepo
from git.exc import GitCommandError
from dvc.repo import Repo as DvcRepo
from dvc.exceptions import DvcException

from config import PROJECT_ROOT, GIT_BRANCH, DVC_REMOTE, GIT_REMOTE


class Config:
    def __init__(self, project_root: str, git_branch:str, git_remote:str, dvc_remote: str):
        # with open(config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        self.project_root = project_root
        self.git_branch = git_branch
        self.dvc_remote = dvc_remote
        self.git_remote = git_remote


class GitManager:
    def __init__(self, repo_path: str):
        self.repo = GitRepo(repo_path)

    def commit(self, message: str) -> bool:
        if self.repo.is_dirty(untracked_files=True):
            self.repo.git.add(A=True)
            self.repo.index.commit(message)
            return True
        return False

    def push(self, remote: str, branch: str, username: str, password: str):
        remote_url = self.repo.remotes[remote].url
        if not remote_url.startswith('https://'):
            remote_url = f'https://github.com/{username}/your-repo.git'
        self.repo.git.push(f'https://{username}:{password}@{remote_url[8:]}', branch)

    def pull(self, remote: str, branch: str, username: str, password: str):
        remote_url = self.repo.remotes[remote].url
        if not remote_url.startswith('https://'):
            remote_url = f'https://github.com/{username}/your-repo.git'
        self.repo.git.pull(f'https://{username}:{password}@{remote_url[8:]}', branch)

    def create_tag(self, tag_name: str):
        self.repo.create_tag(tag_name)

    def push_tag(self, tag_name: str, remote: str):
        self.repo.git.push(remote, tag_name)

    def checkout(self, ref: str):
        self.repo.git.checkout(ref)


class DvcManager:
    def __init__(self, repo_path: str):
        self.repo = DvcRepo(repo_path)

    def init(self):
        if not os.path.exists(os.path.join(self.repo.root_dir, '.dvc')):
            self.repo = DvcRepo.init(self.repo.root_dir)

    def commit(self):
        if self.repo.status():
            self.repo.commit()
            return True
        return False

    def push(self, remote: str):
        self.repo.push(remote=remote)

    def pull(self, remote: str):
        self.repo.pull(remote=remote)

    def checkout(self, targets: Optional[list] = None, force: bool = False):
        self.repo.checkout(targets=targets, force=force)


class DataVersionManager:
    def __init__(self, project_root: str, git_branch: str, git_remote: str, dvc_remote: str, data_path: str):
        self.config = Config(project_root, git_branch, git_remote, dvc_remote)
        self.data_path = data_path
        self.full_data_path = os.path.join(self.config.project_root, data_path)
        self.dvc_file_path = f"{self.full_data_path}.dvc"
        self.git_manager = GitManager(self.config.project_root)
        self.dvc_manager = DvcManager(self.config.project_root)

    def init_dvc(self):
        try:
            self.dvc_manager.init()
            print("DVC initialized")

            if os.path.exists(self.dvc_file_path):
                os.remove(self.dvc_file_path)
                print(f"Removed existing {self.dvc_file_path}")

            self._create_or_update_dvc_yaml()
        except DvcException as e:
            print(f"Error initializing DVC: {e}")

    def _create_or_update_dvc_yaml(self):
        dvc_yaml_path = os.path.join(self.config.project_root, 'dvc.yaml')
        prepare_data_path = os.path.join(self.config.project_root, 'src', 'prepare_data.py')
        dvc_config = {
            'stages': {
                'prepare_data': {
                    'cmd': f'python {os.path.relpath(prepare_data_path, self.config.project_root)}',
                    'deps': [os.path.relpath(prepare_data_path, self.config.project_root)],
                    'outs': [self.data_path]
                }
            }
        }

        with open(dvc_yaml_path, 'w') as f:
            yaml.dump(dvc_config, f)
        print("Updated dvc.yaml file")

    def add_data_to_dvc(self):
        try:
            prepare_data_path = os.path.join(self.config.project_root, 'src', 'prepare_data.py')
            if not os.path.exists(prepare_data_path):
                raise FileNotFoundError(f"prepare_data.py not found: {prepare_data_path}")

            self._create_or_update_dvc_yaml()
            subprocess.run(['dvc', 'repro'], cwd=self.config.project_root, check=True)
            print(f"Updated {self.data_path} in DVC")

            self._update_gitignore()
            self.git_manager.commit(f"Add/Update data in DVC: {self.data_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running dvc repro: {e}")
        except Exception as e:
            print(f"Error adding data to DVC: {e}")

    def _update_gitignore(self):
        gitignore_path = os.path.join(self.config.project_root, '.gitignore')
        data_path_relative = os.path.relpath(self.full_data_path, self.config.project_root)

        if not os.path.exists(gitignore_path):
            with open(gitignore_path, 'w') as f:
                f.write(f"{data_path_relative}\n")
        else:
            with open(gitignore_path, 'r+') as f:
                content = f.read()
                if data_path_relative not in content:
                    f.write(f"\n{data_path_relative}")

        self.git_manager.repo.index.add('.gitignore')

    def create_data_version(self, version_name: str):
        try:
            self.git_manager.repo.git.fetch()
            self.git_manager.checkout(self.config.git_branch)
            self.git_manager.repo.git.pull('origin', self.config.git_branch)

            if self.dvc_manager.commit():
                print("DVC changes committed")
            else:
                print("No DVC changes to commit")

            if self.git_manager.commit(f"Update data version: {version_name}"):
                self.git_manager.create_tag(version_name)
                print(f"Created Git tag for version: {version_name}")
            else:
                print(f"No changes to create version: {version_name}")
                return

            self.git_manager.push_tag(version_name, self.config.git_remote)
            print(f"Pushed tag {version_name} to remote")

        except (DvcException, GitCommandError, Exception) as e:
            print(f"Error creating data version: {e}")

    def push_to_remote(self):
        git_username = input("Enter Git username: ")
        git_password = input("Enter Git password: ")

        try:
            self.git_manager.push(self.config.git_remote, self.config.git_branch, git_username, git_password)
            print("Git changes pushed successfully")

            if self.dvc_manager.commit():
                self.dvc_manager.push(self.config.dvc_remote)
                print("DVC data pushed successfully")
            else:
                print("No DVC data changes to push")
        except (GitCommandError, DvcException, Exception) as e:
            print(f"Error pushing changes: {e}")

    def pull_from_remote(self):
        git_username = input("Enter Git username: ")
        git_password = input("Enter Git password: ")

        try:
            self.git_manager.pull(self.config.git_remote, self.config.git_branch, git_username, git_password)
            print("Git changes pulled successfully")

            self.dvc_manager.pull(self.config.dvc_remote)
            print("DVC data pulled successfully")
        except (GitCommandError, DvcException, Exception) as e:
            print(f"Error pulling changes: {e}")

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

    def append_new_data(self, new_data: pd.DataFrame):
        existing_data = pd.read_csv(self.full_data_path)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        updated_data.to_csv(self.full_data_path, index=False)
        print(f"Appended {len(new_data)} new samples to the dataset.")

    def switch_data_version(self, version_name: str):
        try:
            if version_name not in self.git_manager.repo.tags:
                raise ValueError(f"Version {version_name} does not exist")

            current_branch = self.git_manager.repo.active_branch.name
            self.git_manager.checkout(version_name)
            self.dvc_manager.checkout()
            print(f"Switched to DVC version: {version_name}")

            self.git_manager.checkout(current_branch)
            print(f"Switched back to branch: {current_branch}")

            self.dvc_manager.checkout(targets=[self.data_path], force=True)
            print(f"Updated {self.data_path} to version {version_name}")

        except (GitCommandError, DvcException, Exception) as e:
            print(f"Error switching data version: {e}")

    def create_new_version(self, version_name: str):
        try:
            new_data = self.generate_sample_data()
            self.append_new_data(new_data)
            self.add_data_to_dvc()
            self.create_data_version(version_name)
            self.push_to_remote()
            print(f"New version {version_name} created and pushed successfully.")
        except Exception as e:
            print(f"Error creating new version: {e}")


# Example usage:
if __name__ == "__main__":
    # config_path = "config.yaml"
    data_path = "data/diabetes.csv"
    manager = DataVersionManager(PROJECT_ROOT, GIT_BRANCH, DVC_REMOTE, GIT_REMOTE, data_path)
    manager.init_dvc()
    manager.create_new_version("v1.0")
    manager.switch_data_version("v1.0")
    manager.pull_from_remote()
    manager.push_to_remote()