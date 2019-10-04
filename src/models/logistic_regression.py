import numpy as np
import time

from .classifier import Classifier
from utils.functions import ActivationFunction, sigmoid
from utils.preprocessing import shuffle_data

class LogisticRegression(Classifier):

    def __init__(self, input_size: int, eta=0.0001, activation=sigmoid, batch_size=64, epochs=25):
        # Validate input_size
        if type(input_size) != int:
            raise TypeError('Invalid type for param input_size, expected {}'.format(int))
        if input_size <= 0:
            raise ValueError('Parameter input_size must be positive')

        # Validate eta
        if type(eta) != float:
            raise TypeError('Invalid type for param eta, expected {}'.format(float))
        if eta <= 0:
            raise ValueError('Parameter eta must be positive')

        # Ensure batch_size is > 0
        if type(batch_size) != int:
            raise TypeError('Invalid type for param batch_size, expected {}'.format(int))
        if batch_size <= 0:
            raise ValueError('Parameter batch_size must be positive')

        # Ensure epochs is > 0
        if type(epochs) != int:
            raise TypeError('Invalid type for param epochs, expected {}'.format(int))
        if epochs <= 0:
            raise ValueError('Parameter epochs must be positive')

        self.input_size = input_size
        self.eta = eta
        self.activation = activation
        self.batch_size = batch_size
        self.epochs = epochs

        self.weights = None
        self.randomise()

    def randomise(self):
        """
        Randomly initialise the layer's weights and biases.
        """
        self.weights = np.random.standard_normal((self.input_size + 1,))

    def adjust_weights(self, dw: np.ndarray):
        
        if dw.shape != self.weights.shape:
            raise ValueError('Expected param dw of shape {} but got {}'.format(self.weights.shape, dw.shape))

        self.weights += self.eta * dw

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """
        ones = np.ones((x.shape[0], 1))
        x_tilde = np.hstack((x, ones))
        probability = self.activation.func(x_tilde @ self.weights[:, np.newaxis]).squeeze()
        return np.round(probability).astype('uint8'), probability

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError('Expected 2 unique classes in param y but got {}.'.format(len(classes)))

        y_idxs = y
        y_idxs[y_idxs==classes[0]] = 0
        y_idxs[y_idxs==classes[1]] = 1

        print('Training started')
        
        epoch_idx = 0
        for e in range(self.epochs):

            data, lbl = shuffle_data(x, y_idxs)
            epoch_correct = 0
            epoch_start = time.perf_counter()

            data_split = np.split(data, range(self.batch_size, len(x), self.batch_size))
            lbl_split = np.split(lbl, range(self.batch_size, len(x), self.batch_size))

            batch_idx = 0
            while batch_idx < len(data_split):

                batch = data_split[batch_idx]
                batch_lbls = lbl_split[batch_idx]

                batch_pred_lbls, batch_pred_probs = self.predict(batch)

                batch_correct = (batch_pred_lbls == batch_lbls).sum()

                err = batch_pred_probs - batch_lbls

                # print(batch_pred_probs)
                ones = np.ones((batch.shape[0], 1))
                batch_tilde = np.hstack((batch, ones))
                dw = -np.sum(batch_tilde.T * err, axis=1)
                self.adjust_weights(dw)

                epoch_correct += batch_correct
                batch_idx += 1

            epoch_idx += 1
            epoch_end = time.perf_counter()

            print()
            print('Epoch {} complete in {:.3f}s'.format(e+1, epoch_end - epoch_start))
            print('Accuracy: {:.2f}%'.format(epoch_correct/x.shape[0] * 100))
