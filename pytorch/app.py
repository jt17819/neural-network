from flask import Flask, jsonify, request
import torch
import numpy as np
from predict import main
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        data = request.form

    arr = np.array([1,2,3])
    tensor = torch.from_numpy(arr)
    return jsonify({"message": "Predict from Flask!", "predict": tensor.tolist()})
