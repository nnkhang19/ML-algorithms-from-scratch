from Activation import Activation
class Layer:
    def __init__(self, num_neurons : list, activation : str, reg = 0.0):
        self.num_neurons = num_neurons
        self.activation = Activation(activation)
        self.W = None
        self.b = None
        self.reg = reg

    def forward(self, X):
        if self.W == None and self.b == None:
            shape = (X.shape[1], self.num_neurons)
            self.W = np.random.normal(0, np.sqrt(2 / np.sum(shape)), shape)
            self.b = np.random.normal(0, np.sqrt(1 / self.num_neurons), (1, self.num_neurons))

        self.Z = X @ self.W + self.b
        return self.activation.activate(self.Z)

    def backward(self, X, delta_prev):
       Z = self.Z 
       activation_grad = self.activation.activate_grad(Z)
       delta = delta_prev * activation_grad
       m = X.shape[0]
        
       # regulization is added then reg > 0.0
       W_grad = Z.T @ delta + (self.W * self.reg) / m

       return W_grad, delta
