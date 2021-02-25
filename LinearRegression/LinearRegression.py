import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(object):
    
    def __init__(self, input_shape, learning_rate, iters):
        self.shape = input_shape
        self.learning_rate = learning_rate
        self.iters = iters
        
        # initialize weights:
        self.w = np.random.normal(scale = 1./np.sqrt(np.sum(input_shape)),size = (input_shape[1],1))
        self.b = np.random.normal(scale = 1./np.sqrt(np.sum(input_shape)))
        

    def fit(self, X, y):
        if y.shape[0] == 1:
            y = y.reshape((y.shape[1], y.shape[0]))
        
        for i in range(self.iters):
            y_hat = X @ self.w + self.b
            loss = self.compute_loss(y_hat, y)
            grads = self.compute_gradient(X, y, y_hat)
            
            # update gradient
            self.w -= self.learning_rate * grads
            self.b -= self.learning_rate * loss * 2
            
            if i % 10 == 0:
                print("Iter {}, Loss {}".format(i, loss))

        
    def compute_loss(self, y_hat, y):
        N = y.shape[0]
        return (0.5 / N) * np.sum(y - y_hat)**2

    def compute_gradient(self, X, y, y_hat):
        N = y.shape[0]
        return X.T @ (y_hat - y) / N

    @property
    def weights(self):
        return [self.w, self.b]
    
    def predict(self, X_test):
        return X_test @ self.w + self.b

if __name__ == '__main__':
    X = np.array([[1.0, 2.1], [3.5, 4.6],[7, 8]], dtype = np.float32)
    y = np.array([[2,3,4]])

    model = LinearRegression(X.shape, learning_rate = 0.001, iters = 500)
    model.fit(X, y)
    
