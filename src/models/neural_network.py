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

def th(x: np.array):
    e_nx = np.exp(-x)
    e_x = np.exp(x)
    return (e_x - e_nx)/(e_x + e_nx)

def d_th(x: np.array):
    tanh_x = th(x)
    #print(tanh_x)
    #print(tanh_x**2)

    return 1 - tanh_x**2

sigmoid = ActivationFunction(sig, d_sig)
tanh = ActivationFunction(th, d_th)


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
        self.raw_outputs = None
        self.outputs = None

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

    def get_activations(self, ipt: np.ndarray):
        # No input_shape implies first layer, just return the input as the output
        if self.input_shape is None:
            if ipt.shape != self.output_shape:
                raise ValueError('Invalid shape of ipt array, expected {} but got {}'.format(self.output_shape, ipt.shape))
            self.outputs = ipt
            self.raw_outputs = ipt
            return ipt

        # Ensure input is same shape as self.input_shape
        if ipt.shape != self.input_shape:
            raise ValueError('Invalid shape of ipt array, expected {} but got {}'.format(self.input_shape, ipt.shape))

        ipt = ipt[:, np.newaxis]
        raw_out = self.weights @ ipt + self.biases

        # TODO Consider memory usage here
        self.raw_outputs = raw_out.squeeze()
        self.outputs = self.activation.func(raw_out).squeeze()

        #print(str(min(self.raw_outputs)) + ', ' + str(max(self.raw_outputs)))

        return self.outputs

    def adjust_weights(self, eta: float, prev_outputs: np.ndarray, dC_da: np.ndarray):

        # TODO Validate shapes of errors
        # errors should have dimension 1xn where n is the number of nodes in the layer above
        
        # Compute weight changes using chain rule
        d_sigma = self.activation.d_func(self.raw_outputs)

        dC_dz = dC_da * d_sigma  # a = sigma(z) so da_dz = d_sigma(z)
        dw = eta * (np.outer(dC_dz, prev_outputs))  # prev_outputs = dz_dw
        
        # Compute bias changes
        db = eta * dC_dz
        db = db[:, np.newaxis]

        next_dC_da = np.dot(self.weights.T, d_sigma)

        self.weights += dw
        self.biases += db

        return next_dC_da


class NeuralNetwork(Classifier):

    def __init__(self, layers: list, eta=0.05, batch_size=1):
        # Ensure layers is a ist of NeuralNetworkLayer objects
        for l in layers:
            if not isinstance(l, NeuralNetworkLayer):
                raise TypeError('Invalid type in list {}, expected {}'.format(layers, NeuralNetworkLayer))
        
        # Ensure batch_size is > 0
        if type(batch_size) != int:
            raise TypeError('Invalid batch_size type, expected {}'.format(int))
        if batch_size <= 0:
            raise ValueError('Parameter batch_size must be positive')

        # TODO Validate eta

        self.layers = layers
        self.eta = eta
        self.batch_size = batch_size

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
            ipt = layer.get_activations(ipt)
        return ipt        

    # @staticmethod
    # def cost(predicted: np.ndarray, actual: np.ndarray):
    #     return ((actual - predicted)**2).mean(axis=ax)

    def train(self, x: pd.DataFrame, y: pd.DataFrame):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """
        # TODO implement batch sizes, allowing for batch prediction too
        epochs = 10
        for e in range(epochs):
            print('Epoch: {}'.format(e))

            epoch_correct = 0

            index = 0
            while index < len(x):

                y_pred = self.__predict_single(x[index])
                # TODO Fix assumption that number of output neurons is the same
                # as the number of classes
                y_actual = np.zeros(y_pred.shape)
                y_actual[y[index]] = 1
                dC_da = y_actual - y_pred

                epoch_correct += 1 if np.argmax(y_pred) == y[index] else 0

                # iterate through layers, propagating errors
                layer = len(self.layers) - 1
                while layer > 0:
                    dC_da = self.layers[layer].adjust_weights(self.eta, self.layers[layer-1].outputs, dC_da)
                    layer -= 1

                index += 1

                
            print('Train accuracy: {:.2f}%'.format(epoch_correct/index * 100))

        print('Training complete')

            