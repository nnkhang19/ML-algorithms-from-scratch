import numpy as np


class SVM():
    def __init__(self,regularization_coef = 0.01, learning_rate = 0.001, epochs = 1000):
        self.llambda = regularization_coef
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x_train, y_train):
        n_features = x_train.shape[1]
        n_samples  = x_train.shape[0]

        self.w  = np.random.normal(0, np.sqrt(1/np.sum(x_train.shape)), n_features)
        self.b  = np.sqrt(1/2)

        y_train_conv = np.where(y_train == 0, -1 ,1)

        def hinge(x,y):
            return max(0, 1 - (y*(x @ self.w) + self.b))

        def get_gradient(x, y):
            if hinge(x, y) == 0:
                return (self.llambda * self.w, 0)
            return (self.llambda * self.w - y * x, -y)

        for i in range(self.epochs):
            for j in range(n_samples):
                dw, db = get_gradient(x_train[j], y_train_conv[j])

                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db

    def predict(self,x_test):
        result = x_test @ self.w + self.b
        result = np.where(result < 0 , 0, 1)
        return result

