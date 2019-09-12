import h5py
import numpy as np

from models.classifier import Classifier
from models.neural_network import NeuralNetwork
from models.neural_network import FlatDenseLayer

def main():
    # Load data
    with h5py.File('./data/train/images_training.h5', 'r') as h:
        data_train = np.copy(h['datatrain'])

    with h5py.File('./data/train/labels_training.h5', 'r') as h:
        label_train = np.copy(h['labeltrain'])

    with h5py.File('./data/test/images_testing.h5', 'r') as h:
        data_test_all = np.copy(h['datatest'])

    with h5py.File('./data/test/labels_testing_2000.h5', 'r') as h:
        label_val = np.copy(h['labeltest'])

    data_val = data_test_all[:2000]

    model = NeuralNetwork([
        FlatDenseLayer.from_size(784),
        FlatDenseLayer.from_size(100),
        FlatDenseLayer.from_size(10)
    ])

    model.train(data_train, label_train)
    
    y_pred = model.predict(data_val)
    print('Accuracy: {:.02f}%'.format(100*calc_accuracy(y_pred,label_val)))

def calc_accuracy(y_hat, y):
    if (y_hat.shape == y.shape):
        pred = (y_hat == y)
        return len(np.where(pred == True)[0]) / len(y)

if __name__ == '__main__':
    main()