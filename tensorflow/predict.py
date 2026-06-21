import os
import numpy as np
import tf_keras
from PIL import Image, ImageOps
from io import BytesIO

MODEL_PATH = os.path.join(os.path.dirname(__file__), "digits_model-99.h5")

def predict(raw_data):
    """
    Accepts raw image bytes, returns (predicted_digit, confidence_scores).
    """
    model = tf_keras.models.load_model(MODEL_PATH)

    im = ImageOps.grayscale(Image.open(BytesIO(raw_data)))
    im = im.resize((28, 28), Image.Resampling.LANCZOS)
    im = np.array(im, dtype="float32")
    im = 255.0 - im
    im = im.reshape(1, 28, 28, 1) / 255.0

    scores = model.predict(im, verbose=0)[0]  # shape (10,)
    digit = int(np.argmax(scores))
    return digit, scores.tolist()