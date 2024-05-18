import numpy as np


class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        # self.input = None
        # self.output = None
        # self.num_nodes_in = num_nodes_in
        # self.num_nodes_out = num_nodes_out

        self.weights = np.random.rand(num_nodes_in, num_nodes_out) - 0.5
        # self.biases = np.random.rand(1, num_nodes_out) - 0.5
        self.biases = np.zeros((1,num_nodes_out))

    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.biases

        return self.output

    
    def back_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * output_error

        return input_error

    # def calc_outputs(self, inputs):
    #     # weight_1_1, weight_2_1, weight_1_2, weight_2_2 = 1, 1, 1, 1
    #     activated_inputs = np.zeros(self.num_nodes_out)

    #     for node_out in range(self.num_nodes_out):
    #         weighted_input = self.biases[node_out]
    #         for node_in in range(self.num_nodes_in):
    #             weighted_input += inputs[node_in] * self.weights[node_in, node_out]
    #         activated_inputs[node_out] = self.activation_function(weighted_input)

    #     # output_1 = input_1 * weight_1_1 + input_2 * weight_2_1
    #     # output_2 = input_1 * weight_1_2 + input_2 * weight_2_2

    #     return activated_inputs
    

    # def activation_function(self, weighted_input):
    #     return max(0, weighted_input) #ReLU
    

    # def node_cost(self, output_activation, expected_output):
    #     error = output_activation - expected_output
    #     return error * error


