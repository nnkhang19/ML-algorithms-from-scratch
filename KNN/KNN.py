import numpy as np
from scipy import stats

class KNN():
    def __init__(self, k):
        self.num_of_neighbors = k
        self.predictions = []
        
    def fit(self, x_train, y_train, method = "mahattan_distance"):
        self.x_train = x_train
        self.y_train = y_train
        self.method = method

    def euclidean_distance(self,x_to_classify, x_train):
        distance = (x_to_classify - x_train) ** 2
        distance = np.sum(distance, axis = 1) ** 0.5

        return distance

    def mahattan_distance(self, x_to_classify, x_train):
        distance = np.abs(x_to_classify - x_train)
        distance = np.sum(distance, axis = 1)

        return distance


    def classify(self, x_to_classify):

        if self.method == "mahattan_distance":
            distance = self.mahattan_distance(x_to_classify, self.x_train)
        else:
            distance = self.euclidean_distance(x_to_classify, self.x_train)
        
        index_array = np.argpartition(distance, self.num_of_neighbors)[:self.num_of_neighbors]

        k_nearest_neighbors = self.y_train[index_array]
        
        prediction = stats.mode(k_nearest_neighbors)
            
        return int(prediction[0][0])
        


    def predict(self, x_test):

        for row in x_test:
            prediction = self.classify(row)
            self.predictions.append(prediction)

        return self.predictions
