import numpy as np
import time

from .classifier import Classifier
from utils.functions import ActivationFunction, sigmoid
from utils.preprocessing import shuffle_data

class MultinomialLogisticRegression(Classifier):

    def __init__(self, input_size: int, n_classes: int, eta=0.01, batch_size=64, epochs=25):
        # Validate input_size
        if type(input_size) != int:
            raise TypeError('Invalid type for param input_size, expected {}'.format(int))
        if input_size <= 0:
            raise ValueError('Parameter input_size must be positive')

        # Validate n_classes
        if type(n_classes) != int:
            raise TypeError('Invalid type for param n_classes, expected {}'.format(int))
        if n_classes <= 0:
            raise ValueError('Parameter n_classes must be positive')

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
        self.n_classes = n_classes
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs

        self.weights = None
        self.randomise()

    def randomise(self):
        """
        Randomly initialise the layer's weights and biases.
        """
        self.weights = np.random.standard_normal((self.input_size + 1, self.n_classes))

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
        probabilities = x_tilde @ self.weights
        #print(probabilities)
        return np.argmax(probabilities, axis=1), probabilities

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """

        classes = np.unique(y)
        if len(classes) != self.n_classes:
            raise ValueError('Expected {} unique classes in param y but got {}.'.format(self.n_classes, len(classes)))

        print('Training started')
        
        epoch_idx = 0
        for e in range(self.epochs):

            data, lbl = shuffle_data(x, y)
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
                batch_actual = np.zeros(batch_pred_probs.shape)
                for i in range(batch_actual.shape[1]):
                    batch_actual[batch_lbls.T.squeeze() == i, i] = 1

                # Compute weight changes for weights for each class
                # https://math.stackexchange.com/questions/1428344/what-is-the-derivation-of-the-derivative-of-softmax-regression-or-multinomial-l/2081449
                e_pred = np.exp(batch_pred_probs)
                d_logs = batch_actual - e_pred/np.sum(e_pred, axis=1)[:, np.newaxis]
                ones = np.ones((batch.shape[0], 1))
                batch_tilde = np.hstack((batch, ones))
                dw = (batch_tilde.T @ d_logs)

                self.adjust_weights(dw)

                epoch_correct += batch_correct
                batch_idx += 1

            epoch_idx += 1
            epoch_end = time.perf_counter()

            print()
            print('Epoch {} complete in {:.3f}s'.format(e+1, epoch_end - epoch_start))
            print('Accuracy: {:.2f}%'.format(epoch_correct/x.shape[0] * 100))
