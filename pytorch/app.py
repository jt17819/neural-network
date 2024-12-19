from flask import Flask, jsonify, request
from flask_cors import CORS
import torch, base64
import numpy as np
from predict import main

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.get_json()
        img = base64.b64decode(data["payload"].split(",")[1])
        pred, output = main(img)
        return jsonify({"message": "Predict from Flask!", "ans": pred, "raw": output})

    arr = np.array([1,2,3])
    tensor = torch.from_numpy(arr)
    return jsonify({"message": "Predict from Flask!", "predict": tensor.tolist()})
