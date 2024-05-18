import numpy as np, dill

from neural_net import NeuralNetwork
from layer import Layer
from activations import ActivationLayer


# training data
data_arr = np.random.rand(10000, 1, 2)
label_arr = np.zeros((10000, 1, 2))
# print(data_arr[:,:5])
for i in range(len(label_arr)):
    data = data_arr[i][0]
    check = (data[0] * data[0] + data[1] * data[1]) > 0.4
    label_arr[i][0][0] = check
    label_arr[i][0][1] = not check
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
# data_arr = np.array([[a.tolist()] for a in data_arr])
# label_arr = np.array([[[l]] for l in label_arr])
# print(data_arr, "label",label_arr)
filename = "random.pkl"
# network
nn = NeuralNetwork()
nn.add_layer(Layer(2,4))
nn.add_layer(ActivationLayer("relu"))
nn.add_layer(Layer(4,3))
nn.add_layer(ActivationLayer("relu"))
nn.add_layer(Layer(3,2))
nn.add_layer(ActivationLayer("softmax"))

# # train
nn.set_cost_function("cross_entropy")
nn.train(data_arr, label_arr, epochs=20, learning_rate=0.01, batch_size=100)

with open(filename, "wb") as dill_file:
    dill.dump(nn, dill_file)
    
# with open(filename,"rb") as dill_file:
#     nn = dill.load(dill_file)
# test
x_test = [[[0.1,0.1]],[[0.3,0.3]],[[0.4,0.4]],[[0.5,0.5]],[[0.7,0.7]],[[0.9,0.9]]]
out = nn.predict(x_test)
print(out)