from flask import Flask, jsonify
import torch
import numpy as np
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=["GET", "POST"])
def predict():
    arr = np.array([1,2,3])
    tensor = torch.from_numpy(arr)
    return jsonify({"message": "Predict from Flask!", "predict": tensor.tolist()})
