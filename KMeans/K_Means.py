import numpy as np
class K_Means:
    def __init__(self, n_clusters, epochs = 10000, threshold = 0.001, random_state = 123):
        self.k = n_clusters
        self.epochs = epochs
        self.thresh = threshold
        self.random_state = random_state

    def initialize_centroids(self):
        centroids = self.data[:self.k]
        return centroids

    def update_centroids(self,labels):
        for ki in range(self.k):
            self.centroids[ki] = np.mean(self.data[labels == ki], axis = 0)
        return self.centroids

    def compute_distance(self, distance):
        for ki in range(self.k):
            distance[:,ki] = np.linalg.norm(self.data[:] - self.centroids[ki], axis = 1)
        return distance
    def update_labels(self, distance):
        labels = np.argmin(distance, axis = 1)
        return labels

    def fit(self,data):
        self.data = data
        self.centroids = self.initialize_centroids()
        distances = np.zeros((self.data.shape[0], self.k))
        labels    = None

        for i in range(self.epochs):
            old_centroids = self.centroids
            distances = self.compute_distance(distances)
            labels    = self.update_labels(distances)
            self.centroids = self.update_centroids(labels)

            if np.all(abs(old_centroids - self.centroids) < self.thresh):
                break

    def predict(self, data):
        distances = np.zeros((data.shape[0], self.k))
        for ki in range(self.k):
            distances[:, ki] = np.linalg.norm(data[:] - self.centroids[ki], axis = 1)


        return np.argmin(distances, axis = 1)


        

