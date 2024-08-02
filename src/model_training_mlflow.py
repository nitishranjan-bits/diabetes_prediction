import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_preprocess_data():
    df = pd.read_csv("/home/nitish/Documents/BITS/Semester-3/MLOps/Assignment-1/diabetes_prediction/data/diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def run_experiment(model_type, params):
    mlflow.set_experiment("Diabetes Prediction")
    
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        
        if model_type == "RandomForest":
            model = RandomForestClassifier(**params)
        elif model_type == "KNN":
            model = KNeighborsClassifier(**params)
        else:
            raise ValueError("Unsupported model type")
        
        accuracy, precision, recall, f1 = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model: {model_type}")
        print(f"Parameters: {params}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print("="*50)

if __name__ == "__main__":
    # Run 1: Random Forest with default parameters
    run_experiment("RandomForest", {"n_estimators": 100, "random_state": 42})
    
    # Run 2: Random Forest with more trees
    run_experiment("RandomForest", {"n_estimators": 200, "max_depth": 10, "random_state": 42})
    
    # Run 3: KNN with default parameters
    run_experiment("KNN", {"n_neighbors": 5})
