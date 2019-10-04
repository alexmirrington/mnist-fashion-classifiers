import numpy as np

from .classifier import Classifier
from utils.preprocessing import PCA

class LinearSVM(Classifier):

    # 'alpha' - learning rate of the classifier
    # 'features' - features wanted from feature selection via PCA
    def __init__(self, alpha = 0.01, features = 180):
        self.alpha = alpha
        self.features = features
        return

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of each row of data in x.
        """

        # Select only a subset of features
        print('Performing PCA on test data. This may take a minute or two...')
        new_x, exp_var = PCA(x, self.features)
        w = self.weights[0:len(x),:]
        
        # Predict classes using trained weights
        predicted_y = 0
        for i in range(self.features):
                cur_w = w[:,i].reshape(len(x), 1)
                cur_x = new_x[:,i].reshape(len(x), 1)
                y_pred += cur_w + cur_x

        # Enter predicted classes
        predictions = []
        for i in predicted_y:
            if i > 1:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions

    def train(self, x: np.ndarray, y: np.ndarray, epochs=1):
        """
        Train the classifier on a dataset x and corresponding labels y.
        """

        # Select only a subset of features
        print('Performing PCA on training data. This may take a minute or two...')
        new_x, exp_var = PCA(x, self.features)

        new_y = y.reshape(len(x), 1)
        w = np.zeros((len(x), self.features))

        for e in range(1, epochs+1):
            print('Performing epoch {}...'.format(e))

            # Calculate predicted y from current weight
            cur_y = 0
            for i in range(self.features):
                cur_w = w[:,i].reshape(len(x), 1)
                cur_x = new_x[:,i].reshape(len(x), 1)
                cur_y += cur_w + cur_x

            product = cur_y * new_y

            # Adjust weights
            for i, v in enumerate(product):
                
                # When correctly classified
                if v >= 1:
                    for j in range(self.features):
                        cur_w = w[:,j].reshape(len(x), 1)
                        cur_x = new_x[:,j].reshape(len(x), 1)
                        cur_w = cur_w - self.alpha * (2 * 1 / e * cur_w)
                        w[:,j] = np.squeeze(cur_w)

                # When incorrectly classified
                else:
                    for j in range(self.features):
                        cur_w = w[:,j].reshape(len(x), 1)
                        cur_x = new_x[:,j].reshape(len(x), 1)
                        cur_w = cur_w + self.alpha * (cur_w[i] * new_y[i] - 2 * 1 / e * cur_w)
                        w[:,j] = np.squeeze(cur_w)

        self.weights = w
        print('Successfully trained weights for Linear SVM.')
        return