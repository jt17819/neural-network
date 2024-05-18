import numpy as np
import random


class NeuralNetwork:
    def __init__(self) -> None:
        self.layers = []
        self.cost_function = None
        self.prime_cost_function = None


    def add_layer(self, layer):
        self.layers.append(layer)

    
    def set_cost_function(self, cost_func):
        def cross_entropy(label, output):
            c = 0
            for i in range(len(output)):
                x = output[i] * (output[i] > 0)
                y = label[i]
                # v = -(np.log(x) * (y == 1)) - (np.log(1-x) * (y == 0))
                # v = -(np.log(x[y == 1])) - (np.log(1-x[y == 0]))
                c += np.sum(-(np.log(x[y == 1]))) + np.sum(-(np.log(1-x[y == 0])))
            return c


        def cross_entropy_prime(label, output):
            r = np.zeros_like(label)
            for i in range(len(output)):
                x = output[i]
                x = x[x != 0]
                x = x[x != 1]
                y = label[i]
                # if x != 0 and x != 1:
                r[i] = (y - x) / (x * (x - 1))
            return r
        
        if cost_func == "mean_squared":
            self.cost_function = lambda label, output: np.mean(np.square((label - output)))
            self.prime_cost_function = lambda label, output: 2 * (output - label) / label.size

        elif cost_func == "cross_entropy":
            # TODO implement cross entropy
            self.cost_function = lambda label, output: cross_entropy(label, output) #np.sum(np.negative(np.log(output[label==1] + 1e-9))) + np.sum(np.negative(np.log((1-output[label==0] + 1e-9))))
            self.prime_cost_function = lambda label, output: cross_entropy_prime(label, output) #(label - output) / (output * (output - 1)) * (output != 0) * (output != 1)

        else:
            raise NotImplementedError
        # self.cost_function = cost_func
        # self.prime_cost_function = prime_cost_func

    
    def train(self, full_inputs, full_labels, epochs, learning_rate, batch_size=1):
        # self.learning_rate = learning_rate

        full_sample_size = len(full_inputs)
        for j in range(epochs):
            err = 0
            # print(f"-- EPOCH: {j + 1}/{epochs} --")
            # start batching
            full_inputs, full_labels = self.shuffle(full_inputs, full_labels)
            # print(full_labels[:10])

            start, end = 0, batch_size
            while start < full_sample_size:
                inputs = full_inputs[start:end]
                labels = full_labels[start:end]
                sample_size = len(inputs)
                for i in range(sample_size):
                    output = inputs[i]
                    print(f"Training with {i+1}/{sample_size}", end="\r")
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                        # print(output)
                    
                    err += self.cost_function(labels[i], output)

                    error = self.prime_cost_function(labels[i], output)
                    # print(error)
                    for layer in reversed(self.layers):
                        error = layer.back_propagation(error, learning_rate)
                start += batch_size
                end += batch_size
            # end batching

            err /= full_sample_size
            print(f"\n-- EPOCH {j+1}/{epochs}, Error: {err} --")
        print("\nFinished Training")
        # with open(filename, "wb") as dill_file:
        #     dill.dump(self.layers, dill_file)
            # dill.dump_session(dill_file)
        
    def shuffle(self, a, b):
        tmp = list(zip(a, b))
        random.shuffle(tmp)
        a, b = zip(*tmp)
        return list(a), list(b)

    # def calc_outputs(self, inputs):
    #     for layer in self.layers:
    #         inputs = layer.calc_outputs(inputs)
    #     return inputs


    # def forward_propagation(self, inputs):
    #     self.layers[0].actvaitons = inputs    
    #     for i in range(self.num_layers):
    #         temp = np.add(np.matmul(self.layers[i].activations, self.layers[i].weights), self.layers[i].biases)

    #         match self.layers[i+1].activation_function:
    #             case "sigmoid":
    #                 self.layers[i+1].activations = ac_funcs.sigmoid(temp)
    #             case "softmax":
    #                 self.layers[i+1].activations = ac_funcs.softmax(temp)
    #             case "relu":
    #                 self.layers[i+1].activations = ac_funcs.relu(temp)
    #             case "tanh":
    #                 self.layers[i+1].activations = ac_funcs.tanh(temp)
    #             case _:
    #                 self.layers[i+1].activations = temp

    # def classify(self, inputs):
    #     outputs = self.calc_outputs(inputs)
    #     return self.max_value_index(outputs)
    

    # def max_value_index(values):
    #     return values.index(max(values))
    

    # def cost(self, labels):
    #     match self.cost_function:
    #         case "mean_squared":
    #             self.error += np.mean(np.divide(np.square(np.subtract(labels, self.layers[self.num_layers-1].activations)), 2))

    #         case "cross_entropy":
    #             self.error += np.negative(np.sum(np.multiply(labels, np.log(self.layers[self.num_layers-1].activations))))
            
    #         case _:
    #             print(f"No cost function found for {self.cost_function}, defaulting to mean_squared")
    #             self.error += np.mean(np.divide(np.square(np.subtract(labels, self.layers[self.num_layers-1].activations)), 2))
    

    # def back_propagation(self, labels):
    #     targets= labels
    #     i = self.num_layers - 1
    #     y = self.layers[i].activations
    #     delta_b = np.multiply(y, np.multiply(1-y, targets-y))
    #     delta_w = np.matmul(np.asarray(self.layers[i-1].activations).T, delta_b)
    #     new_weights = self.layers[i-1].weights - self.learning_rate * delta_w
    #     new_bias = self.layers[i-1].biases - self.learning_rate * delta_b

    #     for i in range(i-1, 0, -1):
    #         y = self.layers[i].activations
    #         delta_b = np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_bias, self.layers[i].biases)).T))
    #         delta_w = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_weights, self.layers[i].weights),axis=1).T)))
    #         self.layers[i].weights = new_weights
    #         self.layers[i].biases = new_bias
    #         new_weights = self.layers[i-1].weights - self.learning_rate * delta_w
    #         new_bias = self.layers[i-1].biaes - self.learning_rate * delta_b
            
    #     self.layers[0].weights_for_layer = new_weights
    #     self.layers[0].bias_for_layer = new_bias


    def predict(self, input_data):
        # if filename:
        #     with open(filename,"rb") as dill_file:
        #         self.layers = dill.load(dill_file)
                # dill.load_session(dill_file)
        # print("Layers ", self.layers)
        sample_size = len(input_data)
        result = []

        for i in range(sample_size):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
    

    # def check_accuracy(self, filename, inputs, labels):
    #     dill.load_session(filename)
    #     self.batch_size = len(inputs)
    #     self.forward_propagation(inputs)
    #     a = self.layers[self.num_layers-1].activations
    #     a[np.where(a == np.max(a))] = 1
    #     a[np.where(a != np.max(a))] = 0

    #     total=0
    #     correct=0
    #     for i in range(len(a)):
    #         total += 1
    #         if np.equal(a[i], labels[i]).all():
    #             correct += 1
    #     print(f"Accuracy: {correct*100/total}%")

    #     return