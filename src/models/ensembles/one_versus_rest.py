import numpy as np

from models.classifier import Classifier
from utils.preprocessing import binary_partition_by_class
from copy import deepcopy

class OneVersusRest(Classifier):

    def __init__(self, binary_classifier: Classifier):
        self.classifier = binary_classifier
        self.classifiers = []

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """
        probs = []
        for c in self.classifiers:
            c_classes, c_prob = c.predict(x)
            probs.append(c_prob)

        probs = np.asarray(probs)
        predictions = np.argmax(probs, axis=0)
        return predictions

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """
        self.classifiers = []

        classes = np.unique(y)
        for c in classes:
            classifier_copy = deepcopy(self.classifier)
            data, labels = binary_partition_by_class(x, y, {c,})
            classifier_copy.train(data, labels)
            self.classifiers.append(classifier_copy)


