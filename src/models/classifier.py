from abc import ABC
import pandas as pd

class Classifier(ABC):

    def __init__(self):
        return

    def clear(self):
        """
        Reset all classifier parameters to their defaults.
        """
        return

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the class of each row of data in x.
        """
        return

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """
        return
