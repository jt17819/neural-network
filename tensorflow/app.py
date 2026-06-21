from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
from predict import predict

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return jsonify({"message": "Hello, World!"})

@app.route('/predict', methods=["POST"])
def predict_route():
    data = request.get_json()
    img = base64.b64decode(data["payload"].split(",")[1])
    pred, output = predict(img)
    return jsonify({"message": "Predict from Flask!", "ans": pred, "raw": output})
