import numpy as np, dill

from neural_net import NeuralNetwork
from layer import Layer
from activations import ActivationLayer


# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
# print(x_train, y_train)
# network
nn = NeuralNetwork()
nn.add_layer(Layer(2,3))
nn.add_layer(ActivationLayer("tanh"))
nn.add_layer(Layer(3,1))
nn.add_layer(ActivationLayer("tanh"))

# # train
filename = "xor.pkl"
nn.set_cost_function("mean_squared")
nn.train(x_train, y_train, epochs=1000, learning_rate=0.1, batch_size=4)

# with open(filename, "wb") as dill_file:
#     dill.dump(nn, dill_file)
    
# with open(filename,"rb") as dill_file:
#     nn = dill.load(dill_file)
# # test
out = nn.predict(x_train)
print(out)