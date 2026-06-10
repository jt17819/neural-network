import os
from PIL import Image
from predict import predict

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "digit3-1.png")

with open(IMAGE_PATH, "rb") as f:
    raw_data = f.read()

digit, scores = predict(raw_data)
print(f"Predicted digit: {digit}")
print(f"Confidence scores: {[f'{s:.4f}' for s in scores]}")