import numpy as np, dill
import cv2 as cv

from neural_net import NeuralNetwork
from layer import Layer
from activations import ActivationLayer


# training data
train_X = np.load("Data Set/Digits/Processed Training Images.npy")
train_y = np.load("Data Set/Digits/Processed Training Labels.npy")
test_X = np.load("Data Set/Digits/Processed Test Images.npy")
test_y = np.load("Data Set/Digits/Processed Test Labels.npy")

train_X = train_X.reshape(train_X.shape[0], 1, 28*28)
train_X = train_X.astype('float32')
train_X /= 255
num_classes = 10
train_y = np.eye(num_classes)[train_y]

filename = "processed_mnist.pkl"
# print(train_X.shape)
# print(train_X[0])
# print(train_y.shape)
# print(train_y[0])

# network
nn = NeuralNetwork()
nn.add_layer(Layer(784,200))
nn.add_layer(ActivationLayer("tanh"))
nn.add_layer(Layer(200,100))
nn.add_layer(ActivationLayer("tanh"))
nn.add_layer(Layer(100,10))
nn.add_layer(ActivationLayer("softmax"))

# train
nn.set_cost_function("mean_squared")
nn.train(train_X[:], train_y[:], epochs=20, learning_rate=0.05, batch_size=100)
# nn.train(train_X[:1000], train_y[:1000], epochs=10, learning_rate=0.01, batch_size=100)

with open(filename, "wb") as dill_file:
    dill.dump(nn, dill_file)

# with open(filename,"rb") as dill_file:
#     nn = dill.load(dill_file)

# test
test_X = test_X.reshape(test_X.shape[0], 1, 28*28)
test_X = test_X.astype('float32')
test_X /= 255
correct = 0
test_batch_size = 10000
for i in range(test_batch_size):
    out = nn.predict(test_X[i])
    # print(out)
    # print(np.where(out==np.max(out))[2])
    correct += np.where(out==np.max(out))[2][0] == test_y[i]
# print(test_y[:10])
print(correct / test_batch_size * 100, "%")

img = cv.imread("digit3-1.png")[:,:,0]
img = np.invert([img])
img = img.reshape(img.shape[0], 1, 28*28)
img = img.astype('float32')
img /= 255
p = nn.predict(img)
print(np.argmax(p), p)