from abc import ABC
import numpy as np

class Classifier(ABC):

    def __init__(self):
        return

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """
        return

    def train(self, x: np.ndarray, y: np.ndarray, epochs=25):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """
        return
