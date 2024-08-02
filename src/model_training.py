import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def train_model():
    # Load and preprocess data
    df = pd.read_csv("/home/nitish/Documents/BITS/Semester-3/MLOps/Assignment-1/diabetes_prediction/data/diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Make predictions and calculate accuracy
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)

        # Save model
        mlflow.sklearn.log_model(rf, "random_forest_model")

if __name__ == "__main__":
    train_model()
