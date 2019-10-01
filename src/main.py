import h5py
import numpy as np

from models.classifier import Classifier
from models.naive_bayes import MultinomialNaiveBayes
from models.neural_network import NeuralNetwork,FlatDenseLayer
from utils.functions import sigmoid, tanh, relu, softplus
from utils.metrics import accuracy


def main():
    train_data, train_labels, test_data, test_labels = load_data()

    naive_bayes(train_data, train_labels, test_data, test_labels)
    neural_net(train_data, train_labels, test_data, test_labels)


def naive_bayes(train_data, train_labels, test_data, test_labels):

    # Create and train model
    model = MultinomialNaiveBayes()
    model.train(train_data, train_labels)
    
    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_pred = model.predict(test_data_2k)
    print('Test accuracy: {:.02f}%'.format(100*accuracy(test_pred, test_labels)))

    # Predict 10000 test set samples and save predictions
    test_pred = model.predict(test_data)
    print(len(test_pred))
    save_predictions(naive_bayes.__name__, test_pred)


def neural_net(train_data, train_labels, test_data, test_labels):

    # Create and train model
    model = NeuralNetwork([
        FlatDenseLayer((784,), activation=sigmoid),
        FlatDenseLayer((100,), activation=sigmoid),
        FlatDenseLayer((20,), activation=sigmoid),
        FlatDenseLayer((10,), activation=sigmoid),
    ], eta=0.1, batch_size=64)

    model.train(train_data, train_labels, epochs=50)

    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_activations, test_pred = model.predict(test_data_2k)
    print('Test accuracy: {:.02f}%'.format(100*accuracy(test_pred, test_labels)))

    # Predict 10000 test set samples and save predictions
    test_activations, test_pred = model.predict(test_data)
    print(len(test_pred))
    save_predictions(neural_net.__name__, test_pred)


def load_data():
    with h5py.File('./data/train/images_training.h5', 'r') as h:
        train_data = np.copy(h['datatrain'])

    with h5py.File('./data/train/labels_training.h5', 'r') as h:
        train_labels = np.copy(h['labeltrain'])

    with h5py.File('./data/test/images_testing.h5', 'r') as h:
        test_data = np.copy(h['datatest'])

    with h5py.File('./data/test/labels_testing_2000.h5', 'r') as h:
        test_labels = np.copy(h['labeltest'])

    return train_data, train_labels, test_data, test_labels


def save_predictions(filename, predictions):
    with h5py.File('./results/' + filename + '.h5', 'w') as h:
        h.create_dataset('output', data=predictions)


if __name__ == '__main__':
    main()
