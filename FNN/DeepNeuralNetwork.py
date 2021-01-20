from Layer import Layer

class DNN:
    def __init__(self, learning_rate, num_classes = 2, reg = 1e-5):
        self.layers = []
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.reg = self.reg

    def add_layer(self, num_neurons, activation : str):
        self.layers.append(Layer(num_neurons, activation, self.reg))
    
    def compile(self, loss = 'categorical_cross_entropy', optimizer = 'SGD', epochs = 10000, batch_size = 0):
        assert loss in ['categorical_cross_entropy', 'mean_squared']
        assert optimizer in ['SGD', 'Batch']
        
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, Y_train):
        Y_train_conv = self.__create_one_hot(Y_train)

        self.__train(X_train, Y_train_conv)

    def __train(self, X_train, Y_train):
        if self.optimizer == 'SGD':
            self.__SGD(X_train, Y_train)
        else:
            self.__batch_train(X_train, Y_train)

    def __batch_train(self, X, Y):
        display_step = 100
        all_loss = []

        for e in range(self.epochs):
            outputs = self.__forward_propagate(X_train)
            loss = self.compute_loss(Y_train, outputs[-1])
            grad_list = self.__backward(Y_train, outputs)
            self.update_weights(grad_list)
            
            all_loss.append(loss)

            if (e + 1) % display_step == 0:
                print("Epoch: {}, Loss: {}".format(e + 1, loss))

    def __SGD(self, X_train, Y_train):
        
        """ if batch size > 1, the algorithm is mini batch. """
        
        display_step = 100
        iters = len(X_train) // batch_size
        all_loss = []

        for e in range(epochs):
            indices = np.random.permutation(len(X_train))
            X_train, Y_train = X_train[indices], Y_train[indices]
            loss = 0
            for i in range(iters):
                batch_x = X_train[i * self.batch_size : min((i + 1) * self.batch_size, len(X_train))]
                batch_y = Y_train[i * self.batch_size : min((i + 1) * self.batch_size, len(Y_train))]
                outputs = self.__forward_propagate(batch_x)
                loss += self.compute_loss(batch_y, all_X[-1])
                grad_list = self.__backward_propagate(batch_y, outputs)
                self.update_weight(grad_list)

            if (e + 1) % display_step == 0:
                print("Epoch: {}, Loss: {}".format(e + 1, loss))

    def __create_one_hot(self, labels):
        eye_mat = None
        eye_mat = np.zeros((len(labels), self.num_classes), dtype = np.float64)
        eye_mat[np.arange(len(labels)), labels] = 1.0
        
        return eye_mat


    def __forward_propagate(self, X):
        outputs = [X]
        x = X.copy()
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
            outputs.append(x)

        return outputs

    def compute_loss(self, Y, Y_hat):
        m = len(Y)
        if self.loss == 'categorical_cross_entropy':
            correct_log_probs = np.sum(Y * np.log(Y_hat), axis = 1)
            data_loss = -np.mean(correct_log_probs)
        else:
            correct_log_probs = np.sum((Y - Y_hat)**2, axis = 1)
            data_loss = -np.mean(correct_log_probs)
        
        # add regulization when reg > 0.0
        reg_loss = np.sum(self.reg * self.layers[-1].W) / m
        
        return data_loss + reg_loss

    def __compute_delta_grad_last(self, Y, outputs):
        m = Y.shape[0]
        delta_last = (outputs[-1] - Y) / m
        grad_last  = outputs[-2].T @ delta_last + (self.layers[-1].W + self.reg) / m
        return grad_last, delta_last

    def __backward_propagatate(self, Y, outputs):
        grad_last, delta_prev = self.__compute_delta_grad_last(Y, outputs)
        grad_list = [(grad_last, delta_prev)]
        
        for i in range(len(self.layers) -1 , -1, -1):
            prev_layer = self.layers[i + 1]
            cur_layer = self.layers[i]
            x = outputs[i]
            # calculate delta and grad
            delta_prev = delta_prev @ prev_layer.W.T
            grad_w, delta_prev = layer.backward(x, delta_prev)
            # add to the list to update
            grad_list.append((grad_W, delta_prev))
        
        grad_list = grad_list[::-1]
        return grad_list

    def update_weights(self, momentum_rate = 0.0):
        v = 0.0
        
        for i, layer in enumerate(self.layers):
            grad, delta = grad_list[i]
            v = momentum_rate * v
            layer.W -= v - self.learning_rate * grad
            layer.b -= v - self.learning_rate * delta

    def predict(self, X_test):
        Y_hat = self.forward(X_test)[-1]
        return np.argmax(Y_hat, axis = 1)
