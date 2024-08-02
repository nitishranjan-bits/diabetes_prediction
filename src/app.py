# from flask import Flask, request, jsonify
# import mlflow
# import mlflow.sklearn
# import numpy as np
# from sklearn.preprocessing import StandardScaler
#
# app = Flask(__name__)
#
# # Load the best model
# model = mlflow.sklearn.load_model("mlruns/0/best_model")
#
# # Load the scaler (you'll need to save this during training)
# scaler = StandardScaler()
# scaler.fit(np.load("scaler.npy"))
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     features = np.array(list(data.values())).reshape(1, -1)
#     scaled_features = scaler.transform(features)
#     prediction = model.predict(scaled_features)
#     return jsonify({'prediction': int(prediction[0])})
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')


# app.py

from flask import Flask, request, jsonify
import mlflow.sklearn
import numpy as np

app = Flask(__name__)

# Load the model
model = mlflow.sklearn.load_model("model_for_deployment")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)