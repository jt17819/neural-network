import numpy as np, dill
import cv2 as cv
from PIL import Image


def predict():
    pkl_filename = "processed_mnist.pkl"
    
    with open(pkl_filename,"rb") as dill_file:
        nn = dill.load(dill_file)
    
    img = cv.imread("/tmp/img.jpg")[:,:,0]
    img_pil = Image.fromarray(img)
    img = np.array(img_pil.resize((28, 28), Image.Resampling.LANCZOS))
    img = np.invert([img])
    img = img.reshape(img.shape[0], 1, 28*28)
    img = img.astype('float32')
    img /= 255
    p = nn.predict(img)
    # print(np.argmax(p), p)
    return (np.argmax(p), p)
    
def test():
    pkl_filename = "processed_mnist.pkl"
    
    with open(pkl_filename,"rb") as dill_file:
        nn = dill.load(dill_file)

    img = cv.imread("digit3-1.png")[:,:,0]
    img = np.invert([img])
    img = img.reshape(img.shape[0], 1, 28*28)
    img = img.astype('float32')
    img /= 255
    p = nn.predict(img)
    # print(np.argmax(p), p)
    return p