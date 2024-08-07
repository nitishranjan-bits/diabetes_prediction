import os
from mlops_pipeline import MLOpsPipeline
from data_version_manager import DataVersionManager
from model_factories import RandomForestFactory, LogisticRegressionFactory, SVMFactory
from feature_engineering import StandardScalingStrategy
from config import PROJECT_ROOT, GIT_BRANCH, DVC_REMOTE, GIT_REMOTE


def main():
    data_path = "data/diabetes.csv"
    # data_version_manager = DataVersionManager(PROJECT_ROOT, data_path)

    # Initialize DVC
    # data_version_manager.init_dvc()

    # Create initial version (v1) if it doesn't exist
    # if 'v1' not in data_version_manager.git_repo.tags:
    #     print("Creating initial data version (v1)...")
    #     data_version_manager.create_data_version("v1")

    # if 'v1' not in [tag.name for tag in data_version_manager.git_manager.repo.tags]:
    #     print("Creating initial data version (v1)...")
    #     data_version_manager.create_data_version("v1")

    # Run pipeline with Random Forest on v1
    print("Running pipeline with Random Forest on v1...")
    rf_pipeline = MLOpsPipeline(data_path, RandomForestFactory(), StandardScalingStrategy())
    rf_pipeline.run(data_version="v1")

    # Create new version (v2) with additional data
    print("Creating new data version (v2)...")
    # data_version_manager.create_new_version("v2")

    # Run pipeline with Logistic Regression on v2
    print("Running pipeline with Logistic Regression on v2...")
    lr_pipeline = MLOpsPipeline(data_path, LogisticRegressionFactory(), StandardScalingStrategy())
    lr_pipeline.run(data_version="v2")

    # Run pipeline with SVM on v2
    print("Running pipeline with SVM on v2...")
    svm_pipeline = MLOpsPipeline(data_path, SVMFactory(), StandardScalingStrategy())
    svm_pipeline.run(data_version="v2")

    # Make a prediction using the latest Logistic Regression model
    features = {
        'Pregnancies': 6.0,
        'Glucose': 148.0,
        'BloodPressure': 72.0,
        'SkinThickness': 35.0,
        'Insulin': 0.0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50.0
    }
    prediction = lr_pipeline.predict(features, model_name='diabetes_LogisticRegressionFactory', model_version='latest')
    result = {'prediction': 'Diabetic' if prediction == 1 else 'Non-diabetic'}
    print(f"Prediction for sample features: {result['prediction']}")

    # Push changes to remote repository
    print("Pushing changes to remote repository...")
    # data_version_manager.push_to_remote()

    print("Diabetes prediction pipeline execution completed.")


if __name__ == "__main__":
    main()