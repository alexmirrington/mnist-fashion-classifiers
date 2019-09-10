from abc import ABC
import pandas as pd
import numpy as np
import sys
import random

from .classifier import Classifier


class NeuralNetworkLayer:

    def __init__(self, size):
        # Ensure layer size > 0
        if size <= 0:
            raise ValueError('Invalid layer size, expected size > 0.')

        self.size = size


class FullyConnectedLayer(NeuralNetworkLayer):

    def __init__(self, size: int, prev: NeuralNetworkLayer = None):
        super().__init__(size)
        self.prev = prev
        self.__randomise()

    def __randomise(self):
        """
        Randomly initialise the layer's weights and biases.
        """
        self.biases = np.random.standard_normal(self.size)
        self.weights = np.random.standard_normal((self.size, self.prev.size)) if self.prev is not None else None
        print(self.biases)
        print(self.weights)


class NeuralNetwork(Classifier):

    def __init__(self, shape: tuple):
        # Ensure tuple of ints
        for e in shape:
            if not isinstance(e, int):
                raise TypeError('Invalid type in tuple {}, expected int'.format(shape))
        
        self.layers = []


    def add_layer(self, layer: NeuralNetworkLayer):
        """
        Add a new layer to the network
        """
        layer.prev = self.layers[-1]
        self.layers.append(layer)


    def __randomise(self):
        """
        Randomly initialise the network's weights and biases.
        """
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
