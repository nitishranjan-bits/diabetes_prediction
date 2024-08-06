import optuna
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_and_preprocess_data():
    df = pd.read_csv("/home/nitish/Documents/BITS/Semester-3/MLOps/Assignment-1/diabetes_prediction/data/diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def objective(trial):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def run_optimization():
    mlflow.set_experiment("Diabetes Prediction Hyperparameter Tuning")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    # Log the best parameters and results to MLflow
    with mlflow.start_run():
        mlflow.log_params(trial.params)
        mlflow.log_metric("best_accuracy", trial.value)
        
        # Train the best model
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        best_model = RandomForestClassifier(**trial.params, random_state=42)
        best_model.fit(X_train, y_train)
        
        # Log the best model
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        print(f"Best Model Metrics:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

if __name__ == "__main__":
    run_optimization()
