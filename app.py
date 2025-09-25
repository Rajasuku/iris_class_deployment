# Flask API
import flask
from flask import request, jsonify, Flask
import joblib
import numpy as np

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




if __name__ == '__main__':
    app.run()


