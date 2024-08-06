from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MODEL_REGISTRY_URI

app = Flask(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MODEL_REGISTRY_URI)
client = MlflowClient()


def load_best_model():
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )

    if not runs:
        raise ValueError("No runs found in the experiment")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Loading model from URI: {model_uri}")
    return mlflow.pyfunc.load_model(model_uri)


model = load_best_model()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                         'DiabetesPedigreeFunction', 'Age']
        features = pd.DataFrame([data['features']], columns=feature_names)
        prediction = model.predict(features)[0]
        return jsonify({'prediction': 'Diabetic' if prediction == 1 else 'Non-diabetic'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/model_info', methods=['GET'])
def model_info():
    model_name = model.metadata.get_model_info().name
    latest_version = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])[0]

    return jsonify({
        'name': model_name,
        'version': latest_version.version,
        'stage': latest_version.current_stage,
        'run_id': latest_version.run_id,
        'creation_timestamp': latest_version.creation_timestamp
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
