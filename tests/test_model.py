import unittest
import json
import mlflow
from mlflow.tracking import MlflowClient

from src.app import app


class TestDiabetesApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        test_data = {
            "features": [6, 148, 72, 35, 0, 33.6, 0.627, 50]
        }
        response = self.app.post('/predict',
                                 data=json.dumps(test_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], ['Diabetic', 'Non-diabetic'])

    def test_model_info_endpoint(self):
        response = self.app.get('/model_info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('name', data)
        self.assertIn('version', data)
        self.assertIn('stage', data)
        self.assertIn('run_id', data)
        self.assertIn('creation_timestamp', data)

    def test_mlflow_connection(self):
        client = MlflowClient()
        try:
            experiment = client.get_experiment_by_name("Diabetes Prediction Pipeline")
            self.assertIsNotNone(experiment)
        except mlflow.exceptions.MlflowException as e:
            self.fail(f"Failed to connect to MLflow: {str(e)}")

if __name__ == '__main__':
    unittest.main()
