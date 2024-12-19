from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
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
        data = request.form
        print(data)
        print(type(data))

    arr = np.array([1,2,3])
    tensor = torch.from_numpy(arr)
    return jsonify({"message": "Predict from Flask!", "predict": tensor.tolist()})
