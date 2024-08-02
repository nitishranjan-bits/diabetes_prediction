import os
import subprocess


class DataVersionManager:
    def __init__(self, data_path: str):
        self.data_path = data_path
        # Adjust the project root path to ensure it points to the directory containing .dvc
        self.project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))

    def init_dvc(self):
        # Change to project root to run DVC commands
        os.chdir(self.project_root)
        if not os.path.isdir(".dvc"):
            try:
                subprocess.run(["dvc", "init"], check=True, stderr=subprocess.PIPE)
                print("DVC initialized")
            except subprocess.CalledProcessError as e:
                print(f"Error: {e.stderr.decode()}")
                raise
        else:
            print(".dvc already exists. Skipping initialization.")

    def add_data_to_dvc(self):
        abs_data_path = os.path.abspath(self.data_path)
        relative_data_path = os.path.relpath(abs_data_path, start=self.project_root)

        if not os.path.isfile(abs_data_path):
            raise FileNotFoundError(f"The file {abs_data_path} does not exist.")

        try:
            # Change to project root to run DVC commands
            os.chdir(self.project_root)

            # Run DVC add command
            result = subprocess.run(["dvc", "add", relative_data_path], check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(result.stdout.decode())

            # Run Git add command
            result = subprocess.run(["git", "add", f"{relative_data_path}.dvc"], check=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(result.stdout.decode())

            # Check if .gitignore exists before trying to add it
            gitignore_path = os.path.join(self.project_root, ".gitignore")
            if os.path.isfile(gitignore_path):
                result = subprocess.run(["git", "add", ".gitignore"], check=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                print(result.stdout.decode())
            else:
                print(".gitignore does not exist. Skipping.")

            # Check if there are any changes to commit
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
            if result.stdout.strip():
                # Commit changes
                result = subprocess.run(["git", "commit", "-m", f"Add {relative_data_path} to DVC"], check=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(result.stdout.decode())
                print("Committed changes successfully.")
            else:
                print("No changes to commit.")
        except subprocess.CalledProcessError as e:
            print(f"Error while adding data to DVC: {e.stderr.decode()}")
            print(f"Output: {e.output.decode()}")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

    # def add_data_to_dvc(self):
    #     abs_data_path = os.path.abspath(self.data_path)
    #     relative_data_path = os.path.relpath(abs_data_path, start=self.project_root)
    #
    #     if not os.path.isfile(abs_data_path):
    #         raise FileNotFoundError(f"The file {abs_data_path} does not exist.")
    #
    #     try:
    #         # Change to project root to run DVC commands
    #         os.chdir(self.project_root)
    #         subprocess.run(["dvc", "add", relative_data_path], check=True, stderr=subprocess.PIPE)
    #         subprocess.run(["git", "add", f"{relative_data_path}.dvc"], check=True, stderr=subprocess.PIPE)
    #
    #         # Check if .gitignore exists before trying to add it
    #         gitignore_path = os.path.join(self.project_root, ".gitignore")
    #         if os.path.isfile(gitignore_path):
    #             subprocess.run(["git", "add", ".gitignore"], check=True, stderr=subprocess.PIPE)
    #         else:
    #             print(".gitignore does not exist. Skipping.")
    #
    #         # Check if there are any changes to commit
    #         result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    #         if result.stdout.strip():
    #             # Commit changes
    #             subprocess.run(["git", "commit", "-m", f"Add {relative_data_path} to DVC"], check=True,
    #                            stderr=subprocess.PIPE)
    #             print(f"Added {relative_data_path} to DVC and committed changes.")
    #         else:
    #             print("No changes to commit.")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error while adding data to DVC: {e.stderr.decode()}")

    def create_data_version(self, version_name: str):
        try:
            os.chdir(self.project_root)
            # Commit any changes to DVC files
            subprocess.run(["git", "add", ".dvc"], check=True, stderr=subprocess.PIPE)
            subprocess.run(["git", "commit", "-m", f"Create data version {version_name}"], check=True, stderr=subprocess.PIPE)
            # Tag the commit with the version name
            subprocess.run(["git", "tag", version_name], check=True, stderr=subprocess.PIPE)
            print(f"Created Git tag: {version_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error while creating data version: {e.stderr.decode()}")

    def switch_data_version(self, version_name: str):
        try:
            os.chdir(self.project_root)
            subprocess.run(["git", "checkout", version_name], check=True, stderr=subprocess.PIPE)
            subprocess.run(["dvc", "checkout"], check=True, stderr=subprocess.PIPE)
            print(f"Switched to version: {version_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error while switching data version: {e.stderr.decode()}")
