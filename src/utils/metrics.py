import numpy as np


def accuracy(y_hat, y):
    if (y_hat.shape == y.shape):
        pred = (y_hat == y)
        return len(np.where(pred == True)[0]) / len(y)