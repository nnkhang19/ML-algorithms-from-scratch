import numpy as np
class Layer:
    def __init__(self, num_of_nodes):
        self.nums = num_of_nodes
        self.biases = np.array([[0] * self.nums], dtype = np.float64)
        self.values = np.array([[0] * self.nums], dtype = np.float64)



class FNN:
    def __init__(self,num_of_layers, nodes, learning_rate):
        '''
         num_of_layers : number of layers in the net.
         nodes : a list containing number of nodes in each layer.(e.g : [3, 2, 1])
        
        '''
        self.num_of_layers = num_of_layers
        self.layers = [Layer(num_of_node) for num_of_node in nodes]
        self.weights = self._init_weights()
        self.learning_rate = learning_rate

    def _init_weights(self):
        '''
            TODO: initialize weights

        '''
        weights = []
        for i in range(self.num_of_layers - 1):
            shape = (self.layers[i].nums, self.layers[i + 1].nums)
            weight = np.random.normal(0, np.sqrt(2./np.sum(shape)), shape)
            weights.append(weight)
        return weights

    def forward_propagate(self, x):
        def sigmoid(z):
            return 1. / (1 + np.exp(-z))
        self.layers[0].values = x
        for i in range(1, len(self.layers)):
            self.layers[i].values = sigmoid(self.layers[i-1].values @ self.weights[i - 1] + self.layers[i].biases)

    def backward_propagate(self, y):
        e = y - self.layers[-1].values # (y - y_hat)
        sig = self.layers[-1].values * (1 - self.layers[-1].values) * e
        dW = np.ones((self.layers[-2].nums, self.layers[-1].nums)) * self.layers[-2].values.T
        tmp = np.ones((self.layers[-2].nums, self.layers[-1].nums)) * sig
        dW = dW * tmp
        self.weights[-1] += self.learning_rate * dW
        self.layers[-1].biases += self.learning_rate * sig

        for i in range(self.num_of_layers - 2, 0, -1):
            sig = self.layers[i].values * (1 - self.layers[i].values) * (sig @ self.weights[i].T)
            dW = np.ones((self.layers[i-1].nums, self.layers[i].nums)) * self.layers[i-1].values.T
            tmp = np.ones((self.layers[i-1].nums, self.layers[i].nums)) * sig
            dW = dW * tmp
            self.weights[i-1] += self.learning_rate * dW
            self.layers[i].biases += self.learning_rate * sig

    def output(self):
        '''
            return y_hat values in the rightmost layer
        '''
        return self.layers[-1].values
