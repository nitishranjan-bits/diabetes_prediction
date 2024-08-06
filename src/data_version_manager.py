# import os
# import subprocess
# import dvc.api
# import yaml
# from dvc.repo import Repo
# from dvc.exceptions import DvcException
# from git import Repo as GitRepo
# from git.exc import GitCommandError
# import pandas as pd
# import numpy as np
# from config import GIT_USERNAME, GIT_PASSWORD, PROJECT_ROOT, GIT_BRANCH
#
#
# class DataVersionManager:
#     def __init__(self, data_path: str):
#         self.data_path = data_path
#         self.project_root = PROJECT_ROOT
#         self.full_data_path = os.path.join(self.project_root, data_path)
#         self.dvc_file_path = f"{self.full_data_path}.dvc"
#         self.git_repo = GitRepo(self.project_root)
#         self.dvc_repo = Repo(self.project_root)
#         self.prepare_data_path = os.path.join(self.project_root, 'src', 'prepare_data.py')
#
#     def init_dvc(self):
#         try:
#             if not os.path.exists(os.path.join(self.project_root, '.dvc')):
#                 self.dvc_repo = Repo.init(self.project_root)
#                 print("DVC initialized")
#             else:
#                 print("DVC is already initialized")
#
#             if os.path.exists(self.dvc_file_path):
#                 os.remove(self.dvc_file_path)
#                 print(f"Removed existing {self.dvc_file_path}")
#
#             self._create_or_update_dvc_yaml()
#         except DvcException as e:
#             print(f"Error initializing DVC: {e}")
#
#     def _create_or_update_dvc_yaml(self):
#         dvc_yaml_path = os.path.join(self.project_root, 'dvc.yaml')
#         dvc_config = {
#             'stages': {
#                 'prepare_data': {
#                     'cmd': f'python {os.path.relpath(self.prepare_data_path, self.project_root)}',
#                     'deps': [os.path.relpath(self.prepare_data_path, self.project_root)],
#                     'outs': [self.data_path]
#                 }
#             }
#         }
#
#         with open(dvc_yaml_path, 'w') as f:
#             yaml.dump(dvc_config, f)
#         print("Updated dvc.yaml file")
#
#     def add_data_to_dvc(self):
#         try:
#             if not os.path.exists(self.prepare_data_path):
#                 raise FileNotFoundError(f"prepare_data.py not found: {self.prepare_data_path}")
#
#             self._create_or_update_dvc_yaml()
#             subprocess.run(['dvc', 'repro'], cwd=self.project_root, check=True)
#             print(f"Updated {self.data_path} in DVC")
#
#             self._update_gitignore()
#             self._commit_changes(f"Add/Update data in DVC: {self.data_path}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error running dvc repro: {e}")
#         except Exception as e:
#             print(f"Error adding data to DVC: {e}")
#
#     def _update_gitignore(self):
#         gitignore_path = os.path.join(self.project_root, '.gitignore')
#         data_path_relative = os.path.relpath(self.full_data_path, self.project_root)
#
#         if not os.path.exists(gitignore_path):
#             with open(gitignore_path, 'w') as f:
#                 f.write(f"{data_path_relative}\n")
#         else:
#             with open(gitignore_path, 'r+') as f:
#                 content = f.read()
#                 if data_path_relative not in content:
#                     f.write(f"\n{data_path_relative}")
#
#         self.git_repo.index.add('.gitignore')
#
#     def create_data_version(self, version_name: str, feature_branch: str):
#         try:
#             # Fetch updates from remote
#             self.git_repo.git.fetch()
#
#             # Check if the feature branch exists locally, if not create it and track the remote feature branch
#             if feature_branch not in self.git_repo.branches:
#                 self.git_repo.git.checkout('-b', feature_branch, f'origin/{feature_branch}')
#             else:
#                 # Checkout the feature branch
#                 self.git_repo.git.checkout(feature_branch)
#
#             # Ensure the local feature branch is up to date with the remote feature branch
#             self.git_repo.git.pull('origin', feature_branch)
#             if self.dvc_repo.status():
#                 self.dvc_repo.commit()
#                 print("DVC changes committed")
#             else:
#                 print("No DVC changes to commit")
#
#             if self._commit_changes(f"Update data version: {version_name}"):
#                 self.git_repo.create_tag(version_name)
#                 print(f"Created Git tag for version: {version_name}")
#             else:
#                 print(f"No changes to create version: {version_name}")
#                 return
#
#             try:
#                 self.git_repo.git.push('origin', version_name, force=True)
#                 print(f"Pushed tag {version_name} to remote")
#             except GitCommandError as e:
#                 print(f"Git error pushing tag {version_name}: {e}")
#
#         except DvcException as e:
#             print(f"DVC error creating version: {e}")
#         except GitCommandError as e:
#             print(f"Git error creating version: {e}")
#         except Exception as e:
#             print(f"Error creating data version: {e}")
#
#     def _commit_changes(self, message: str):
#         try:
#             if self.git_repo.is_dirty(untracked_files=True):
#                 self.git_repo.git.add(A=True)
#                 self.git_repo.index.commit(message)
#                 print(f"Changes committed: {message}")
#                 return True
#             else:
#                 print("No changes to commit")
#                 return False
#         except GitCommandError as e:
#             print(f"Git error committing changes: {e}")
#             return False
#
#     def push_to_remote(self, remote_name: str = 'origin', feature_branch: str = 'feature'):
#         try:
#             # Fetch updates from remote
#             self.git_repo.git.fetch(remote_name)
#
#             # Ensure the feature branch is checked out and up to date
#             if feature_branch not in self.git_repo.branches:
#                 self.git_repo.git.checkout('-b', feature_branch, f'{remote_name}/{feature_branch}')
#             else:
#                 self.git_repo.git.checkout(feature_branch)
#                 self.git_repo.git.pull(remote_name, feature_branch)
#
#             # Check for changes to push
#             if not self.git_repo.is_dirty() and not self.git_repo.untracked_files:
#                 local_commits = list(self.git_repo.iter_commits(f'{remote_name}/{feature_branch}..{feature_branch}'))
#                 if not local_commits:
#                     print("No changes to push")
#                     return
#
#             # Push Git changes to the remote feature branch
#             remote_url = self.git_repo.remotes[remote_name].url
#             if not remote_url.startswith('https://'):
#                 remote_url = f'https://github.com/{GIT_USERNAME}/your-repo.git'
#             self.git_repo.git.push(f'https://{GIT_USERNAME}:{GIT_PASSWORD}@{remote_url[8:]}', feature_branch)
#             print("Git changes pushed successfully")
#
#             # Push DVC changes
#             dvc_remote = input("Enter DVC remote name (default 'origin'): ") or 'origin'
#             if self.dvc_repo.status():
#                 self.dvc_repo.push(remote=dvc_remote)
#                 print("DVC data pushed successfully")
#             else:
#                 print("No DVC data changes to push")
#         except GitCommandError as e:
#             print(f"Git error pushing changes: {e}")
#         except DvcException as e:
#             print(f"DVC error pushing data: {e}")
#         except Exception as e:
#             print(f"Error pushing changes: {e}")
#
#     # def push_to_remote(self, remote_name: str = 'origin'):
#     #     try:
#     #         if not self.git_repo.is_dirty() and not self.git_repo.untracked_files:
#     #             local_commits = list(self.git_repo.iter_commits('origin/master..master'))
#     #             if not local_commits:
#     #                 print("No changes to push")
#     #                 return
#     #
#     #         remote_url = self.git_repo.remotes[remote_name].url
#     #         if not remote_url.startswith('https://'):
#     #             remote_url = f'https://github.com/{GIT_USERNAME}/your-repo.git'
#     #         self.git_repo.git.push(f'https://{GIT_USERNAME}:{GIT_PASSWORD}@{remote_url[8:]}', 'master')
#     #         print("Git changes pushed successfully")
#     #
#     #         dvc_remote = input("Enter DVC remote name (default 'origin'): ") or 'origin'
#     #         if self.dvc_repo.status():
#     #             self.dvc_repo.push(remote=dvc_remote)
#     #             print("DVC data pushed successfully")
#     #         else:
#     #             print("No DVC data changes to push")
#     #     except GitCommandError as e:
#     #         print(f"Git error pushing changes: {e}")
#     #     except DvcException as e:
#     #         print(f"DVC error pushing data: {e}")
#     #     except Exception as e:
#     #         print(f"Error pushing changes: {e}")
#
#     def pull_from_remote(self, remote_name: str = 'origin'):
#         try:
#             remote_url = self.git_repo.remotes[remote_name].url
#             if not remote_url.startswith('https://'):
#                 remote_url = f'https://github.com/{GIT_USERNAME}/your-repo.git'
#             self.git_repo.git.pull(f'https://{GIT_USERNAME}:{GIT_PASSWORD}@{remote_url[8:]}', 'master')
#             print("Git changes pulled successfully")
#
#             dvc_remote = input("Enter DVC remote name (default 'origin'): ") or 'origin'
#             self.dvc_repo.pull(remote=dvc_remote)
#             print("DVC data pulled successfully")
#         except GitCommandError as e:
#             print(f"Git error pulling changes: {e}")
#         except DvcException as e:
#             print(f"DVC error pulling data: {e}")
#         except Exception as e:
#             print(f"Error pulling changes: {e}")
#
#     def generate_sample_data(self, n_samples: int = 500):
#         columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
#                    'DiabetesPedigreeFunction', 'Age', 'Outcome']
#
#         data = pd.DataFrame({
#             'Pregnancies': np.random.randint(0, 17, n_samples),
#             'Glucose': np.random.randint(70, 200, n_samples),
#             'BloodPressure': np.random.randint(40, 120, n_samples),
#             'SkinThickness': np.random.randint(0, 60, n_samples),
#             'Insulin': np.random.randint(0, 500, n_samples),
#             'BMI': np.random.uniform(18, 50, n_samples).round(1),
#             'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples).round(3),
#             'Age': np.random.randint(21, 80, n_samples),
#             'Outcome': np.random.randint(0, 2, n_samples)
#         })
#
#         return data
#
#     def append_new_data(self, new_data: pd.DataFrame):
#         existing_data = pd.read_csv(self.full_data_path)
#         updated_data = pd.concat([existing_data, new_data], ignore_index=True)
#         updated_data.to_csv(self.full_data_path, index=False)
#         print(f"Appended {len(new_data)} new samples to the dataset.")
#
#     def switch_data_version(self, version_name: str):
#         try:
#             if version_name not in self.git_repo.tags:
#                 raise ValueError(f"Version {version_name} does not exist")
#
#             current_branch = self.git_repo.active_branch.name
#             self.git_repo.git.checkout(version_name)
#             self.dvc_repo.checkout()
#             print(f"Switched to DVC version: {version_name}")
#
#             self.git_repo.git.checkout(current_branch)
#             print(f"Switched back to branch: {current_branch}")
#
#             self.dvc_repo.checkout(targets=[self.data_path], force=True)
#             print(f"Updated {self.data_path} to version {version_name}")
#
#         except GitCommandError as e:
#             print(f"Git error switching version: {e}")
#         except DvcException as e:
#             print(f"DVC error switching version: {e}")
#         except Exception as e:
#             print(f"Error switching data version: {e}")
#
#     def create_new_version(self, version_name: str):
#         try:
#             new_data = self.generate_sample_data()
#             self.append_new_data(new_data)
#             self.add_data_to_dvc()
#             self.create_data_version(version_name, GIT_BRANCH)
#             self.push_to_remote()
#             print(f"New version {version_name} created and pushed successfully.")
#         except Exception as e:
#             print(f"Error creating new version: {e}")


import os
import subprocess
import dvc.api
import yaml
from dvc.repo import Repo
from dvc.exceptions import DvcException
from git import Repo as GitRepo
from git.exc import GitCommandError
import pandas as pd
import numpy as np
from config import PROJECT_ROOT, GIT_BRANCH


class DataVersionManager:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.project_root = PROJECT_ROOT
        self.full_data_path = os.path.join(self.project_root, data_path)
        self.dvc_file_path = f"{self.full_data_path}.dvc"
        self.git_repo = GitRepo(self.project_root)
        self.dvc_repo = Repo(self.project_root)
        self.prepare_data_path = os.path.join(self.project_root, 'src', 'prepare_data.py')

    def init_dvc(self):
        try:
            if not os.path.exists(os.path.join(self.project_root, '.dvc')):
                self.dvc_repo = Repo.init(self.project_root)
                print("DVC initialized")
            else:
                print("DVC is already initialized")

            if os.path.exists(self.dvc_file_path):
                os.remove(self.dvc_file_path)
                print(f"Removed existing {self.dvc_file_path}")

            self._create_or_update_dvc_yaml()
        except DvcException as e:
            print(f"Error initializing DVC: {e}")

    def _create_or_update_dvc_yaml(self):
        dvc_yaml_path = os.path.join(self.project_root, 'dvc.yaml')
        dvc_config = {
            'stages': {
                'prepare_data': {
                    'cmd': f'python {os.path.relpath(self.prepare_data_path, self.project_root)}',
                    'deps': [os.path.relpath(self.prepare_data_path, self.project_root)],
                    'outs': [self.data_path]
                }
            }
        }

        with open(dvc_yaml_path, 'w') as f:
            yaml.dump(dvc_config, f)
        print("Updated dvc.yaml file")

    def add_data_to_dvc(self):
        try:
            if not os.path.exists(self.prepare_data_path):
                raise FileNotFoundError(f"prepare_data.py not found: {self.prepare_data_path}")

            self._create_or_update_dvc_yaml()
            subprocess.run(['dvc', 'repro'], cwd=self.project_root, check=True)
            print(f"Updated {self.data_path} in DVC")

            self._update_gitignore()
            self._commit_changes(f"Add/Update data in DVC: {self.data_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running dvc repro: {e}")
        except Exception as e:
            print(f"Error adding data to DVC: {e}")

    def _update_gitignore(self):
        gitignore_path = os.path.join(self.project_root, '.gitignore')
        data_path_relative = os.path.relpath(self.full_data_path, self.project_root)

        if not os.path.exists(gitignore_path):
            with open(gitignore_path, 'w') as f:
                f.write(f"{data_path_relative}\n")
        else:
            with open(gitignore_path, 'r+') as f:
                content = f.read()
                if data_path_relative not in content:
                    f.write(f"\n{data_path_relative}")

        self.git_repo.index.add('.gitignore')

    def create_data_version(self, version_name: str, feature_branch: str):
        try:
            # Fetch updates from remote
            self.git_repo.git.fetch()

            # Check if the feature branch exists locally, if not create it and track the remote feature branch
            if feature_branch not in self.git_repo.branches:
                self.git_repo.git.checkout('-b', feature_branch, f'origin/{feature_branch}')
            else:
                # Checkout the feature branch
                self.git_repo.git.checkout(feature_branch)

            # Ensure the local feature branch is up to date with the remote feature branch
            self.git_repo.git.pull('origin', feature_branch)
            if self.dvc_repo.status():
                self.dvc_repo.commit()
                print("DVC changes committed")
            else:
                print("No DVC changes to commit")

            if self._commit_changes(f"Update data version: {version_name}"):
                self.git_repo.create_tag(version_name)
                print(f"Created Git tag for version: {version_name}")
            else:
                print(f"No changes to create version: {version_name}")
                return

            try:
                self.git_repo.git.push('origin', version_name, force=True)
                print(f"Pushed tag {version_name} to remote")
            except GitCommandError as e:
                print(f"Git error pushing tag {version_name}: {e}")

        except DvcException as e:
            print(f"DVC error creating version: {e}")
        except GitCommandError as e:
            print(f"Git error creating version: {e}")
        except Exception as e:
            print(f"Error creating data version: {e}")

    def _commit_changes(self, message: str):
        try:
            if self.git_repo.is_dirty(untracked_files=True):
                self.git_repo.git.add(A=True)
                self.git_repo.index.commit(message)
                print(f"Changes committed: {message}")
                return True
            else:
                print("No changes to commit")
                return False
        except GitCommandError as e:
            print(f"Git error committing changes: {e}")
            return False

    def push_to_remote(self, remote_name: str = 'origin', feature_branch: str = 'feature'):
        git_username = input("Enter Git username: ")
        git_password = input("Enter Git password: ")

        try:
            # Fetch updates from remote
            self.git_repo.git.fetch(remote_name)

            # Ensure the feature branch is checked out and up to date
            if feature_branch not in self.git_repo.branches:
                self.git_repo.git.checkout('-b', feature_branch, f'{remote_name}/{feature_branch}')
            else:
                self.git_repo.git.checkout(feature_branch)
                self.git_repo.git.pull(remote_name, feature_branch)

            # Check for changes to push
            if not self.git_repo.is_dirty() and not self.git_repo.untracked_files:
                local_commits = list(self.git_repo.iter_commits(f'{remote_name}/{feature_branch}..{feature_branch}'))
                if not local_commits:
                    print("No changes to push")
                    return

            # Push Git changes to the remote feature branch
            remote_url = self.git_repo.remotes[remote_name].url
            if not remote_url.startswith('https://'):
                remote_url = f'https://github.com/{git_username}/your-repo.git'
            self.git_repo.git.push(f'https://{git_username}:{git_password}@{remote_url[8:]}', feature_branch)
            print("Git changes pushed successfully")

            # Push DVC changes
            dvc_remote = input("Enter DVC remote name (default 'origin'): ") or 'origin'
            if self.dvc_repo.status():
                self.dvc_repo.push(remote=dvc_remote)
                print("DVC data pushed successfully")
            else:
                print("No DVC data changes to push")
        except GitCommandError as e:
            print(f"Git error pushing changes: {e}")
        except DvcException as e:
            print(f"DVC error pushing data: {e}")
        except Exception as e:
            print(f"Error pushing changes: {e}")

    def pull_from_remote(self, remote_name: str = 'origin'):
        git_username = input("Enter Git username: ")
        git_password = input("Enter Git password: ")

        try:
            remote_url = self.git_repo.remotes[remote_name].url
            if not remote_url.startswith('https://'):
                remote_url = f'https://github.com/{git_username}/your-repo.git'
            self.git_repo.git.pull(f'https://{git_username}:{git_password}@{remote_url[8:]}', 'master')
            print("Git changes pulled successfully")

            dvc_remote = input("Enter DVC remote name (default 'origin'): ") or 'origin'
            self.dvc_repo.pull(remote=dvc_remote)
            print("DVC data pulled successfully")
        except GitCommandError as e:
            print(f"Git error pulling changes: {e}")
        except DvcException as e:
            print(f"DVC error pulling data: {e}")
        except Exception as e:
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
            if version_name not in self.git_repo.tags:
                raise ValueError(f"Version {version_name} does not exist")

            current_branch = self.git_repo.active_branch.name
            self.git_repo.git.checkout(version_name)
            self.dvc_repo.checkout()
            print(f"Switched to DVC version: {version_name}")

            # Switch back to the original branch
            self.git_repo.git.checkout(current_branch)
            print(f"Switched back to branch: {current_branch}")

            # Update the data file to the version-specific state
            self.dvc_repo.checkout(targets=[self.data_path], force=True)
            print(f"Updated {self.data_path} to version {version_name}")

        except GitCommandError as e:
            print(f"Git error switching version: {e}")
        except DvcException as e:
            print(f"DVC error switching version: {e}")
        except Exception as e:
            print(f"Error switching data version: {e}")

    def create_new_version(self, version_name: str):
        try:
            new_data = self.generate_sample_data()
            self.append_new_data(new_data)
            self.add_data_to_dvc()
            self.create_data_version(version_name, GIT_BRANCH)
            self.push_to_remote()
            print(f"New version {version_name} created and pushed successfully.")
        except Exception as e:
            print(f"Error creating new version: {e}")
