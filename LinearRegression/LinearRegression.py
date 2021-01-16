import numpy as np

class LinearRegression:
    def __init__(self, _shape, epochs = 10000, learning_rate = 0.001):
        self.num_of_samples = _shape[0]
        self.dim = _shape[1]
        self.w = np.random.normal((0, np.sqrt(1/np.sum(_shape))),size = (self.dim + 1, 1))
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.costs = []
        y = y.reshape((-1, 1))
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)

        for e in range(self.epochs):
            y_hat = X @ self.w
            loss = (np.sum(y_hat - y) ** 2) / (2 * self.num_of_samples)
            dw   = (X.T @ (y_hat - y))/ float(self.num_of_samples)
            self.w = self.w - self.learning_rate * dw
            if e % 1000 == 0:
                print("Epoch {} Loss: {}".format(e, loss))
                self.costs.append(loss)

    def predict(self, data):
        data = np.concatenate((data, np.ones((data.shape[0], 1))), axis = 1)
        return (data @ self.w).reshape((len(data), ))

