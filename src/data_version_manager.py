import os
import subprocess
import dvc.api
from dvc.repo import Repo
from dvc.exceptions import DvcException
from git import Repo as GitRepo
from git.exc import GitCommandError
import getpass

class DataVersionManager:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.full_data_path = os.path.join(self.project_root, data_path)
        self.dvc_file_path = f"{self.full_data_path}.dvc"
        self.git_repo = GitRepo(self.project_root)
        self.dvc_repo = Repo(self.project_root)

    def init_dvc(self):
        try:
            if not os.path.exists(os.path.join(self.project_root, '.dvc')):
                self.dvc_repo = Repo.init(self.project_root)
                print("DVC initialized")
            else:
                print("DVC is already initialized")
        except DvcException as e:
            print(f"Error initializing DVC: {e}")

    def add_data_to_dvc(self):
        try:
            if not os.path.exists(self.full_data_path):
                raise FileNotFoundError(f"Data file not found: {self.full_data_path}")

            if os.path.exists(self.dvc_file_path):
                print(f"DVC file already exists for {self.data_path}. Checking for changes...")
                status = self.dvc_repo.status([self.data_path])
                if status.get(self.data_path, {}).get('changed'):
                    self.dvc_repo.add(self.data_path)
                    print(f"Updated {self.data_path} in DVC")
                else:
                    print(f"No changes detected in {self.data_path}")
            else:
                self.dvc_repo.add(self.data_path)
                print(f"Added {self.data_path} to DVC")

            self._update_gitignore()
            self._commit_changes(f"Add/Update data in DVC: {self.data_path}")
        except DvcException as e:
            print(f"DVC error adding data: {e}")
        except GitCommandError as e:
            print(f"Git error adding data: {e}")
        except Exception as e:
            print(f"Error adding data to DVC: {e}")
    # def add_data_to_dvc(self):
    #     try:
    #         if not os.path.exists(self.full_data_path):
    #             raise FileNotFoundError(f"Data file not found: {self.full_data_path}")
    #
    #         if os.path.exists(self.dvc_file_path):
    #             print(f"DVC file already exists for {self.data_path}. Checking for changes...")
    #             if self.dvc_repo.status([self.data_path])[self.data_path].get('changed'):
    #                 self.dvc_repo.add(self.data_path)
    #                 print(f"Updated {self.data_path} in DVC")
    #             else:
    #                 print(f"No changes detected in {self.data_path}")
    #         else:
    #             self.dvc_repo.add(self.data_path)
    #             print(f"Added {self.data_path} to DVC")
    #
    #         self._update_gitignore()
    #         self._commit_changes(f"Add/Update data in DVC: {self.data_path}")
    #     except DvcException as e:
    #         print(f"DVC error adding data: {e}")
    #     except GitCommandError as e:
    #         print(f"Git error adding data: {e}")
    #     except Exception as e:
    #         print(f"Error adding data to DVC: {e}")

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

    def create_data_version(self, version_name: str):
        try:
            if self.dvc_repo.status():
                self.dvc_repo.commit()
                print("DVC changes committed")
            else:
                print("No DVC changes to commit")

            if self._commit_changes(f"Update data version: {version_name}"):
                self.git_repo.create_tag(version_name)
                print(f"Created DVC version: {version_name}")
            else:
                print(f"No changes to create version: {version_name}")
        except DvcException as e:
            print(f"DVC error creating version: {e}")
        except GitCommandError as e:
            print(f"Git error creating version: {e}")
        except Exception as e:
            print(f"Error creating data version: {e}")

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
        except GitCommandError as e:
            print(f"Git error switching version: {e}")
        except DvcException as e:
            print(f"DVC error switching version: {e}")
        except Exception as e:
            print(f"Error switching data version: {e}")

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

    def push_to_remote(self, remote_name: str = 'origin'):
        try:
            # Check if there are changes to push
            if not self.git_repo.is_dirty() and not self.git_repo.untracked_files:
                local_commits = list(self.git_repo.iter_commits('origin/master..master'))
                if not local_commits:
                    print("No changes to push")
                    return

            # Push Git changes
            username = input("Enter Git username: ")
            password = getpass.getpass("Enter Git password: ")
            remote_url = self.git_repo.remotes[remote_name].url
            if not remote_url.startswith('https://'):
                remote_url = f'https://github.com/{username}/your-repo.git'
            self.git_repo.git.push(f'https://{username}:{password}@{remote_url[8:]}', 'master')
            print("Git changes pushed successfully")

            # Push DVC data
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
        try:
            # Pull Git changes
            username = input("Enter Git username: ")
            password = getpass.getpass("Enter Git password: ")
            remote_url = self.git_repo.remotes[remote_name].url
            if not remote_url.startswith('https://'):
                remote_url = f'https://github.com/{username}/your-repo.git'
            self.git_repo.git.pull(f'https://{username}:{password}@{remote_url[8:]}', 'master')
            print("Git changes pulled successfully")

            # Pull DVC data
            dvc_remote = input("Enter DVC remote name (default 'origin'): ") or 'origin'
            self.dvc_repo.pull(remote=dvc_remote)
            print("DVC data pulled successfully")
        except GitCommandError as e:
            print(f"Git error pulling changes: {e}")
        except DvcException as e:
            print(f"DVC error pulling data: {e}")
        except Exception as e:
            print(f"Error pulling changes: {e}")
# import os
# import subprocess
#
#
# class DataVersionManager:
#     def __init__(self, data_path: str):
#         self.data_path = data_path
#         # Adjust the project root path to ensure it points to the directory containing .dvc
#         self.project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
#
#     def init_dvc(self):
#         # Change to project root to run DVC commands
#         os.chdir(self.project_root)
#         if not os.path.isdir(".dvc"):
#             try:
#                 subprocess.run(["dvc", "init"], check=True, stderr=subprocess.PIPE)
#                 print("DVC initialized")
#             except subprocess.CalledProcessError as e:
#                 print(f"Error: {e.stderr.decode()}")
#                 raise
#         else:
#             print(".dvc already exists. Skipping initialization.")
#
#     def add_data_to_dvc(self):
#         abs_data_path = os.path.abspath(self.data_path)
#         relative_data_path = os.path.relpath(abs_data_path, start=self.project_root)
#
#         if not os.path.isfile(abs_data_path):
#             raise FileNotFoundError(f"The file {abs_data_path} does not exist.")
#
#         try:
#             # Change to project root to run DVC commands
#             os.chdir(self.project_root)
#
#             # Run DVC add command
#             result = subprocess.run(["dvc", "add", relative_data_path], check=True, stdout=subprocess.PIPE,
#                                     stderr=subprocess.PIPE)
#             print(result.stdout.decode())
#
#             # Run Git add command
#             result = subprocess.run(["git", "add", f"{relative_data_path}.dvc"], check=True, stdout=subprocess.PIPE,
#                                     stderr=subprocess.PIPE)
#             print(result.stdout.decode())
#
#             # Check if .gitignore exists before trying to add it
#             gitignore_path = os.path.join(self.project_root, ".gitignore")
#             if os.path.isfile(gitignore_path):
#                 result = subprocess.run(["git", "add", ".gitignore"], check=True, stdout=subprocess.PIPE,
#                                         stderr=subprocess.PIPE)
#                 print(result.stdout.decode())
#             else:
#                 print(".gitignore does not exist. Skipping.")
#
#             # Check if there are any changes to commit
#             result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
#             if result.stdout.strip():
#                 # Commit changes
#                 result = subprocess.run(["git", "commit", "-m", f"Add {relative_data_path} to DVC"], check=True,
#                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#                 print(result.stdout.decode())
#                 print("Committed changes successfully.")
#             else:
#                 print("No changes to commit.")
#         except subprocess.CalledProcessError as e:
#             print(f"Error while adding data to DVC: {e.stderr.decode()}")
#             print(f"Output: {e.output.decode()}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {str(e)}")
#
#     # def add_data_to_dvc(self):
#     #     abs_data_path = os.path.abspath(self.data_path)
#     #     relative_data_path = os.path.relpath(abs_data_path, start=self.project_root)
#     #
#     #     if not os.path.isfile(abs_data_path):
#     #         raise FileNotFoundError(f"The file {abs_data_path} does not exist.")
#     #
#     #     try:
#     #         # Change to project root to run DVC commands
#     #         os.chdir(self.project_root)
#     #         subprocess.run(["dvc", "add", relative_data_path], check=True, stderr=subprocess.PIPE)
#     #         subprocess.run(["git", "add", f"{relative_data_path}.dvc"], check=True, stderr=subprocess.PIPE)
#     #
#     #         # Check if .gitignore exists before trying to add it
#     #         gitignore_path = os.path.join(self.project_root, ".gitignore")
#     #         if os.path.isfile(gitignore_path):
#     #             subprocess.run(["git", "add", ".gitignore"], check=True, stderr=subprocess.PIPE)
#     #         else:
#     #             print(".gitignore does not exist. Skipping.")
#     #
#     #         # Check if there are any changes to commit
#     #         result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
#     #         if result.stdout.strip():
#     #             # Commit changes
#     #             subprocess.run(["git", "commit", "-m", f"Add {relative_data_path} to DVC"], check=True,
#     #                            stderr=subprocess.PIPE)
#     #             print(f"Added {relative_data_path} to DVC and committed changes.")
#     #         else:
#     #             print("No changes to commit.")
#     #     except subprocess.CalledProcessError as e:
#     #         print(f"Error while adding data to DVC: {e.stderr.decode()}")
#
#     # def create_data_version(self, version_name: str):
#     #     try:
#     #         os.chdir(self.project_root)
#     #         # Commit any changes to DVC files
#     #         subprocess.run(["git", "add", ".dvc"], check=True, stderr=subprocess.PIPE)
#     #         subprocess.run(["git", "commit", "-m", f"Create data version {version_name}"], check=True, stderr=subprocess.PIPE)
#     #         # Tag the commit with the version name
#     #         subprocess.run(["git", "tag", version_name], check=True, stderr=subprocess.PIPE)
#     #         print(f"Created Git tag: {version_name}")
#     #     except subprocess.CalledProcessError as e:
#     #         print(f"Error while creating data version: {e.stderr.decode()}")
#
#     def create_data_version(self, version_name: str):
#         try:
#             os.chdir(self.project_root)
#
#             # Stage changes related to DVC files
#             result = subprocess.run(["git", "add", ".dvc"], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#             print(f"Staged changes: {result.stdout.decode()}")
#
#             # Commit changes with a message
#             result = subprocess.run(["git", "commit", "-m", f"Create data version {version_name}"], check=True,
#                                     stderr=subprocess.PIPE, stdout=subprocess.PIPE)
#             print(f"Commit output: {result.stdout.decode()}")
#
#             # Tag the commit with the version name
#             result = subprocess.run(["git", "tag", version_name], check=True, stderr=subprocess.PIPE,
#                                     stdout=subprocess.PIPE)
#             print(f"Created Git tag: {version_name}")
#             print(f"Tag output: {result.stdout.decode()}")
#
#         except subprocess.CalledProcessError as e:
#             print(f"Error while creating data version: {e.stderr.decode()}")
#
#     def switch_data_version(self, version_name: str):
#         try:
#             os.chdir(self.project_root)
#             subprocess.run(["git", "checkout", version_name], check=True, stderr=subprocess.PIPE)
#             subprocess.run(["dvc", "checkout"], check=True, stderr=subprocess.PIPE)
#             print(f"Switched to version: {version_name}")
#         except subprocess.CalledProcessError as e:
#             print(f"Error while switching data version: {e.stderr.decode()}")
