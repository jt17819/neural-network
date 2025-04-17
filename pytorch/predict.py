import torch
import numpy as np
from model import Model
import cv2 as cv
from PIL import Image, ImageOps
from io import BytesIO

def main(raw_data):
    device = torch.device('cpu')
    # print(device)

    model = Model().to(device)
    model.load_state_dict(torch.load("94-mnist-pytorch.pt", weights_only=True))
    model.eval()
    
    im = ImageOps.grayscale(Image.open(BytesIO(raw_data)))
    im = np.array(im.resize((28, 28), Image.Resampling.LANCZOS))
    im = np.invert([im])
    im = im.reshape(im.shape[0], 1, 28*28)
    im = im.astype("float32")
    im /= 255

    # model = Model().to(device)
    # print(img)
    with torch.no_grad():
        # data = data.to(device)
        data = torch.from_numpy(im)
        output = model.forward(data)
    pred = output.max(2)[1] # get the index of the max log-

    return pred[0][0].item(), output.tolist()[0]


def test():
    device = torch.device('cpu')
    print(device)

    model = Model().to(device)
    model.load_state_dict(torch.load("94-mnist-pytorch.pt", weights_only=True))

    img = cv.imread("digit3-1.png")[:,:,0]
    img = np.invert([img])
    img = img.reshape(img.shape[0], 1, 28*28)
    img = img.astype('float32')
    img /= 255
    
    
    with torch.no_grad():
        # data = data.to(device)
        data = torch.from_numpy(img)
        output = model.forward(data)
    pred = output.max(2)[1] # get the index of the max log-
    print(pred[0][0].item(), output.tolist()[0])

    # print(np.argmax(p), p)
    return pred


if __name__ == "__main__":
    test()
