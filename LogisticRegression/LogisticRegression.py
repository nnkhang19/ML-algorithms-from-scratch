import numpy as np

class LogisticRegression():
    def __init__(self, shape, epochs = 10000, learning_rate = 0.01):
        '''
            shape is the size of X_train (e.g : (100, 2))
        '''
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = np.random.normal(0, np.sqrt(2./np.sum(shape)), size = (shape[1] + 1, 1));

    def _sigmoid(self, x):
        z = x @ self.w
        result = 1./(1 + np.exp(-z))
        return result

    def feed_forward(self, x):
        y_hat = self._sigmoid(x)
        return y_hat

    def fit(self, x_train,  y_train, batch_size : int):
        '''
            - Data should be normalized before passed in, since exp(-x) tends to zero when x is too large or sigmoid value can be overflow
              when x is too negatively small.
            - Mini-batch gradient descent is applied.
            - X_train is expanded by 1 column of 1s.
            - y_train is reshaped.
        '''
        x_train = np.concatenate((x_train, np.ones((x_train.shape[0],1))), axis = 1)
        y_train = y_train.reshape(-1,1)
        self.errors = []
        train_errors = float("inf")
        iters = len(x_train) // batch_size
        for e in range(self.epochs):
            for i in range(iters):
                batch_x = x_train[i * batch_size : min((i + 1) * batch_size, len(x_train))]
                batch_y = y_train[i * batch_size : min((i + 1) * batch_size, len(y_train))]
                y_hat = self.feed_forward(batch_x)
                loss  = self.get_loss(y_hat, batch_y)
                dw    = self.get_gradient(batch_x, y_hat, batch_y)
                self.w -= dw * self.learning_rate
                if train_errors > loss:
                    train_errors = loss
            self.errors += [train_errors]

            if e % 100 == 0:
                print("Epochs : {}, Loss: {}".format(e, train_errors))

    def get_loss(self,y_hat, y_train):
        loss = -np.log(y_hat) *  y_train -(1 - y_train) *np.log(1 - y_hat)
        loss = np.sum(loss)
        N    = len(y_hat)

        return (1./N) * loss
    
    def get_gradient(self, x_train, y_hat, y_train):
        N    = len(y_train)
        dw   = (1./N) * (x_train.T @ (y_hat - y_train))
        return dw

    def predict(self, data):
        data = np.concatenate((data, np.ones((data.shape[0],1))), axis = 1)
        predictions = data @ self.w
        predictions = np.where(predictions >= 0.5, 1,0)
        return predictions.reshape((len(data),))

