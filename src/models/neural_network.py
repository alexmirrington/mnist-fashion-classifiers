from abc import ABC
import pandas as pd
import numpy as np
import math
import sys
import random
import time

from .classifier import Classifier
from utils.preprocessing import shuffle_data


class ActivationFunction:

    def __init__(self, func: callable, d_func: callable):
        self.func = func
        self.d_func = d_func

def f_sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))

def df_sigmoid(x: np.array):
    sig_x = f_sigmoid(x)
    return sig_x * (1 - sig_x)

def f_tanh(x: np.array):
    e_nx = np.exp(-x)
    e_x = np.exp(x)
    return (e_x - e_nx)/(e_x + e_nx)

def df_tanh(x: np.array):
    tanh_x = f_tanh(x)
    return 1 - tanh_x**2

def f_relu(x: np.array):
    x_copy = np.copy(x)
    x_copy[x_copy<0] = 0
    return x_copy

def df_relu(x: np.array):
    ret = (x > 0).astype(int)
    return ret

sigmoid = ActivationFunction(f_sigmoid, df_sigmoid)
tanh = ActivationFunction(f_tanh, df_tanh)
relu = ActivationFunction(f_relu, df_relu)

def sse(predicted: np.ndarray, actual: np.ndarray):
    return 0.5*(actual - predicted)**2

def d_sse(predicted: np.ndarray, actual: np.ndarray):
    return actual - predicted


class NeuralNetworkLayer:

    def __init__(self, output_shape: tuple, activation=sigmoid):
        # Ensure shapes are valid
        NeuralNetworkLayer.__validate_shape(output_shape)
        self.output_shape = output_shape
        self.input_shape = None
        self.activation = activation

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
        super().__init__(output_shape, activation)
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

    @staticmethod
    def __get_valid_batch(array: np.ndarray, shape: tuple):
        """
        Given a specified shape (s0, s1, ..., sk), verify that the array
        is of shape (s0, s1, ..., sk, n).
        """
        if len(array.shape) > len(shape) + 1:
            raise ValueError('Too many dimensions in param array.')

        if len(array.shape) < len(shape):
            raise ValueError('Not enough dimensions in param array.')
        
        # Check if array has shape (s0, s1, ..., sk)
        # If so, cast to shape (s0, s1, ..., sk, 1)
        if len(array.shape) == len(shape):
            array = array[:, np.newaxis]

        if array.shape[:-1] != shape:
            raise ValueError('Invalid shape of param array, expected {} but got {}'.format(shape, array.shape))
        
        return array

    def get_activations(self, ipt: np.ndarray):
        """
        Get the activations/outputs of the current layer, given the input array.
        """

        # No input_shape implies first layer, just return the input as the output
        if self.input_shape is None:
            ipt = FlatDenseLayer.__get_valid_batch(ipt, self.output_shape)
            
            self.outputs = ipt
            self.raw_outputs = ipt

            return ipt

        ipt = FlatDenseLayer.__get_valid_batch(ipt, self.input_shape)

        raw_out = self.weights @ ipt + self.biases

        # TODO Consider memory usage here
        self.raw_outputs = raw_out.squeeze()
        self.outputs = self.activation.func(raw_out).squeeze()

        # print(str(min(self.raw_outputs)) + ', ' + str(max(self.raw_outputs)))

        return self.outputs

    def adjust_weights(self, eta: float, prev_outputs: np.ndarray, dC_da: np.ndarray):
        # print('dC_da:', dC_da.shape)
        # print('prev_out:', prev_outputs.shape)
        # TODO Validate shapes of errors
        
        # Compute weight changes using chain rule
        d_sigma = self.activation.d_func(self.raw_outputs)

        dC_dz = dC_da * d_sigma  # a = sigma(z) so da_dz = d_sigma(z)

        dw = eta * (dC_dz @ prev_outputs.T)  # prev_outputs = dz_dw
        #print(dw)

        # Compute bias changes
        #print(dC_dz.sum(axis=1).shape)
        db = eta * dC_dz.sum(axis=1)
        db = db[:, np.newaxis]

        next_dC_da = np.dot(self.weights.T, dC_dz)

        self.weights += dw
        self.biases += db

        return next_dC_da


class NeuralNetwork(Classifier):

    def __init__(self, layers: list, eta=0.05, batch_size=64):
        # Ensure layers is a ist of NeuralNetworkLayer objects
        for l in layers:
            if not isinstance(l, NeuralNetworkLayer):
                raise TypeError('Invalid type in list {}, expected {}'.format(layers, NeuralNetworkLayer))
        
        # Ensure batch_size is > 0
        if type(batch_size) != int:
            raise TypeError('Invalid type for param batch_size, expected {}'.format(int))
        if batch_size <= 0:
            raise ValueError('Parameter batch_size must be positive')

        # Validate eta
        if type(eta) != float:
            raise TypeError('Invalid type for param eta, expected {}'.format(float))
        if eta <= 0:
            raise ValueError('Parameter eta must be positive')
        
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """

        # TODO: Ensure x has same dimension as input layer
        activation = x.T
        for layer in self.layers:
            activation = layer.get_activations(activation)
        return activation, np.argmax(activation, axis=0)
        
    def __predict_single(self, row):
        ipt = row
        for layer in self.layers:
            ipt = layer.get_activations(ipt)
        return ipt        


    def train(self, x: np.ndarray, y: np.ndarray, epochs=25):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """

        print('Training started')
        
        epoch_idx = 0
        for e in range(epochs):

            data, lbl = shuffle_data(x, y)

            epoch_correct = 0
            epoch_start = time.perf_counter()

            data_split = np.split(data, range(self.batch_size, len(x), self.batch_size))
            lbl_split = np.split(lbl, range(self.batch_size, len(x), self.batch_size))

            batch_idx = 0
            while batch_idx < len(data_split):

                batch = data_split[batch_idx]
                batch_lbls = lbl_split[batch_idx]

                batch_pred, batch_pred_lbls = self.predict(batch)

                batch_actual = np.zeros(batch_pred.shape)

                for i in range(batch_actual.shape[0]):
                    batch_actual[i, batch_lbls.T.squeeze() == i] = 1

                
                dC_da = d_sse(batch_pred, batch_actual)

                batch_correct = (batch_pred_lbls == batch_lbls.squeeze()).sum()

                # iterate through layers, propagating errors
                layer = len(self.layers) - 1
                while layer > 0:
                    dC_da = self.layers[layer].adjust_weights(self.eta, self.layers[layer-1].outputs, dC_da)
                    layer -= 1

                epoch_correct += batch_correct
                batch_idx += 1

            epoch_idx += 1
            epoch_end = time.perf_counter()

            print('Accuracy: {:.2f}%'.format(epoch_correct/x.shape[0] * 100))
            print('Epoch {} complete in {:.3f}s'.format(e+1, epoch_end - epoch_start))
            print()

        print('Training complete')

            