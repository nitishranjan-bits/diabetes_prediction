# diabetes_prediction

mlflow server --host 0.0.0.0 --port 5000

mlflow server --default-artifact-root ./mlflow_artifacts --host 0.0.0.0 --port 5000


curl -X POST http://0.0.0.0:5001/predict \
     -H "Content-Type: application/json" \
     -d '{
           "features": {
             "Pregnancies": 6.0,
             "Glucose": 148.0,
             "BloodPressure": 72.0,
             "SkinThickness": 35.0,
             "Insulin": 0.0,
             "BMI": 33.6,
             "DiabetesPedigreeFunction": 0.627,
             "Age": 50.0
           }
         }'

