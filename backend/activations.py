import numpy as np


class ActivationLayer:
    def __init__(self, activation_func):
        if activation_func == "sigmoid":
                self.activation = self.sigmoid
                self.activation_prime = self.sigmoid_prime
        elif activation_func == "softmax":
                self.activation = self.softmax
                self.activation_prime = self.softmax_prime
        elif activation_func == "relu":
                self.activation = self.relu
                self.activation_prime = self.relu_prime
        elif activation_func == "tanh":
                self.activation = self.tanh
                self.activation_prime = self.tanh_prime
        else:
            raise NotImplementedError


    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        
        return self.output
    
    
    def back_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
     

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(np.negative(layer))))


    def sigmoid_prime(self, layer):
        f = np.add(1, np.exp(np.negative(layer)))
        return np.divide(np.exp(np.negative(layer)), np.multiply(f, f))


    def relu(self, layer):
        layer[layer < 0] = 0
        return layer


    def relu_prime(self, layer):
        layer[layer < 0] = 0
        layer[layer > 0] = 1
        return layer


    def softmax(self, layer):
        ex = np.exp(layer)
        # print("LAYER ",layer)
        if isinstance(layer[0], np.ndarray):
            # print("SOFTMAX", ex / np.sum(ex, axis=1, keepdims=True))
            return ex / np.sum(ex, axis=1, keepdims=True)
        else:
            return ex / np.sum(ex, keepdims=True)


    def softmax_prime(self, layer):
        ex = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            exp_sum = np.sum(ex, axis=1, keepdims=True)
            return (ex * exp_sum - ex * ex) / (exp_sum * exp_sum)
        else:
            exp_sum = np.sum(ex, keepdims=True) 
            return (ex * exp_sum - ex * ex) / (exp_sum * exp_sum)


    def tanh(self, layer):
        return np.tanh(layer)


    def tanh_prime(self, layer):
        return 1 - (np.tanh(layer) * np.tanh(layer))
