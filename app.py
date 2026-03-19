# Flask API
import flask
from flask import request, jsonify, Flask
import joblib
import numpy as np
import os

app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello, World!'

model = joblib.load('iris_classifier.pkl')
@app.route('/iris', methods=['POST'])
def iris():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    class_names = ['setosa', 'versicolor', 'virginica']
    result = {"prediction": class_names[prediction]}
    return jsonify(result)
port = int(os.environ.get("PORT", 8501))
 



if __name__ == '__main__':
    app.run()


