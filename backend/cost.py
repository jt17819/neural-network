import numpy as np


class Cost:
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
    
    # match cost_func:
    #     case "mean_squared":
    #         self.cost_function = lambda label, output: np.mean(np.square((label - output)))
    #         self.prime_cost_function = lambda label, output: 2 * (output - label) / label.size

    #     case "cross_entropy":
    #         # TODO implement cross entropy
    #         self.cost_function = lambda label, output: cross_entropy(label, output) #np.sum(np.negative(np.log(output[label==1] + 1e-9))) + np.sum(np.negative(np.log((1-output[label==0] + 1e-9))))
    #         self.prime_cost_function = lambda label, output: cross_entropy_prime(label, output) #(label - output) / (output * (output - 1)) * (output != 0) * (output != 1)

    #     case _:
    #         raise NotImplementedError
    # self.cost_function = cost_func
    # self.prime_cost_function = prime_cost_func