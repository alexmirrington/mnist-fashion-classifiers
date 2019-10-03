import numpy as np
import time

from .classifier import Classifier
from utils.functions import euclidean_squared


class NearestNeighbour(Classifier):

    def __init__(self, k: int, dist: callable=euclidean_squared):
        if k <= 0:
            raise ValueError('Parameter k must be greater than 0.')

        self.k = k
        self.data = None
        self.classes = None
        self.dist = dist

    def clear(self):
        """
        Reset all classifier parameters to their defaults.
        """
        self.data = None
        self.classes = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """
        if x.shape[1] != self.data.shape[1]:
            raise ValueError('Invalid dimension of param x, expected {} but got {}'.format(self.data.shape[1], x.shape[1]))
        
        start = time.perf_counter()
        predictions = np.apply_along_axis(self.__predict_single, 1, x)
        end = time.perf_counter()
        print('Predicted {} samples in {}s'.format(x.shape[0], end - start))
        return predictions

    def __predict_single(self, x: np.ndarray) -> np.ndarray:
        dst = self.dist(self.data, x, axis=1)
        k_smallest = np.argpartition(dst, self.k)[:self.k]
        k_classes = self.classes[k_smallest]
        return np.argmax(np.bincount(k_classes))


    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the classifier on a dataset x and corresponding labels y.
        K-Nearest-Neighbour is a lazy learning algorithm, so we simply store
        the data for prediction later.
        """
        self.data = x
        self.classes = y
