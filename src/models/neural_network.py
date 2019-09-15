from abc import ABC
import pandas as pd
import numpy as np
import math
import sys
import random

from .classifier import Classifier


class ActivationFunction:

    def __init__(self, func: callable, d_func: callable):
        self.func = func
        self.d_func = d_func

def sig(x: np.array):
    return 1 / (1 + np.exp(-x))

def d_sig(x: np.array):
    sig_x = sig(x)
    return sig_x * (1 - sig_x)

sigmoid = ActivationFunction(sig, d_sig)



class NeuralNetworkLayer:

    def __init__(self, output_shape: tuple):
        # Ensure shapes are valid
        NeuralNetworkLayer.__validate_shape(output_shape)
        self.output_shape = output_shape
        self.input_shape = None

    def set_input_shape(self, shape: tuple):
        NeuralNetworkLayer.__validate_shape(shape)
        self.input_shape = shape

    @staticmethod
    def __validate_shape(shape: tuple):
        for val in shape:
            if not isinstance(val, int):
                raise TypeError('Invalid shape, expected a tuple of ints')
            if val <= 0:
                raise ValueError('Invalid value in shape, expected > 0.')
        

class FlatDenseLayer(NeuralNetworkLayer):

    def __init__(self, output_shape: tuple, activation=sigmoid):
        super().__init__(output_shape)
        self.activation = activation
        
    @classmethod
    def from_size(cls, size: int):
        return cls((size,))

    def randomise(self):
        """
        Randomly initialise the layer's weights and biases.
        """
        self.biases = None
        self.weights = None

        if self.input_shape is not None:
            self.biases = np.random.standard_normal((self.output_shape[0], 1))
            self.weights = np.random.standard_normal((self.output_shape[0], self.input_shape[0]))

    def get_output(self, ipt: np.ndarray):
        # No input_shape implies first layer, just return the input as the output
        if self.input_shape is None:
            if ipt.shape != self.output_shape:
                raise ValueError('Invalid shape of ipt array, expected {} but got {}'.format(self.output_shape, ipt.shape))
            return ipt

        # Ensure input is same shape as self.input_shape
        if ipt.shape != self.input_shape:
            raise ValueError('Invalid shape of ipt array, expected {} but got {}'.format(self.input_shape, ipt.shape))

        ipt = ipt[:, np.newaxis]
        ret = self.activation.func(self.weights @ ipt + self.biases)
        ret = ret.squeeze()

        return ret


class NeuralNetwork(Classifier):

    def __init__(self, layers: list):
        # Ensure layers is a ist of NeuralNetworkLayer objects
        for l in layers:
            if not isinstance(l, NeuralNetworkLayer):
                raise TypeError('Invalid type in list {}, expected {}'.format(layers, NeuralNetworkLayer))
        
        self.layers = layers

        # Link layers
        i = 1
        while i < len(self.layers):
            self.layers[i].set_input_shape(self.layers[i-1].output_shape)
            i += 1
        
        # Randomise weights
        self.__randomise()

    def __randomise(self):
        """
        Randomly initialise the network's weights and biases.
        """
        for layer in self.layers:
            layer.randomise()

    def clear(self):
        """
        Reset all classifier parameters to their defaults.
        """
        return

    def predict(self, x) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """
        # TODO: Ensure x has same dimension as input layer
        # TODO: Make more efficient

        x = np.apply_along_axis(self.__predict_single, 1, x)
        return np.argmax(x, axis=1)
        
    def __predict_single(self, row):
        ipt = row
        for layer in self.layers:
            ipt = layer.get_output(ipt)
        return ipt

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """
        # TODO: Implement this
        return
        
