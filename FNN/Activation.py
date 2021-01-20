import numpy as np

class Activation:
    def __init__(self, func):
        assert func in ['sigmoid', 'relu', 'tanh', 'softmax']
        self.func = func
        self.reg  = reg

    def __sigmoid(self, x):
        return 1./(1 + np.exp(-x))
    
    def __sigmoid_grad(self, x):
        g = self.__sigmoid(x)
        return g * ( 1 - g)

    def __relu(self, x):
        """ Input : A numpy array x """
        return np.where(x > 0, x, 0)
    
    def __relu_grad(self, x):
        return np.where(x > 0, 1, 0)

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def __tanh_grad(self, x):
        tanh = self.__tanh(x)
        return 1 - tanh**2

    def __softmax(self, x):
        z_prime = np.amax(x, axis = 1, keepdims = True)
        z_prime = np.exp(x - z_prime)
        probs = np.sum(z_prime, axis = 1, keepdims = True)
        probs = z_prime / probs
        return probs

    def activate(self, x):
        if self.func == 'sigmoid':
            return self.__sigmoid(x)
        elif self.func == 'relu':
            return self.__relu(x)
        elif self.func == 'tanh':
            return self.__tanh(x)
        else: 
            return self.__softmax(x)

    def activate_grad(self, x):
        if self.func == 'sigmoid':
            return self.__sigmoid_grad(x)
        elif self.func == 'relu':
            return self.__relu_grad(x)
        elif self.func == 'tanh':
            return self.__tanh_grad(x)
        else: 
            return self.__softmax_grad(x)

