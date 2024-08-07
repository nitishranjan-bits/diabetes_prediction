import os

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "Diabetes Prediction Pipeline"
# MLFLOW_TRACKING_URI = os.path.join(PROJECT_ROOT, 'mlruns')
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_ARTIFACT_STORE = os.path.join(PROJECT_ROOT, 'mlflow_artifacts')
MODEL_REGISTRY_URI = f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}"

# Ensure necessary directories exist
# os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)
os.makedirs(MLFLOW_ARTIFACT_STORE, exist_ok=True)

# Set MLflow configuration
os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
os.environ['MLFLOW_REGISTRY_URI'] = MODEL_REGISTRY_URI
os.environ['MLFLOW_ARTIFACT_URI'] = MLFLOW_ARTIFACT_STORE

GIT_BRANCH = "develop"
DVC_REMOTE = "origin"
GIT_REMOTE = "origin"
