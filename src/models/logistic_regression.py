import numpy as np
from .classifier import Classifier
from utils.functions import ActivationFunction, sigmoid


class LogisticRegression(Classifier):

    def __init__(self, input_size: int, activation=sigmoid, batch_size=64, epochs=25):
        if input_size <= 0:
            raise ValueError('Parameter input_size must be positive')
        if batch_size <= 0:
            raise ValueError('Parameter batch_size must be positive')

        self.input_size = input_size
        self.batch_size = batch_size
        self.activation = activation

        self.weights = None
        self.randomise()

    def randomise(self):
        """
        Randomly initialise the layer's weights and biases.
        """
        self.weights = np.random.standard_normal((self.input_size + 1, 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """
        ones = np.ones((x.shape[0], 1))
        x_tilde = np.hstack((x, ones))
        probability = self.activation.func(x_tilde @ self.weights).squeeze()
        return np.round(probability).astype('uint8'), probability

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """
        return
