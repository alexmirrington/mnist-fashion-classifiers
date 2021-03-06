import h5py
import numpy as np

from models.classifier import Classifier
from models.naive_bayes import MultinomialNaiveBayes
from models.nearest_neighbour import NearestNeighbour
from models.logistic_regression import LogisticRegression
from models.multinomial_logistic_regression import MultinomialLogisticRegression
from models.neural_network import NeuralNetwork,FlatDenseLayer
from models.ensembles.one_versus_rest import OneVersusRest
from models.linear_svm import LinearSVM
from utils.functions import sigmoid, tanh, relu, softplus
from utils.functions import manhattan
from utils.metrics import accuracy, confusion_matrix, recall, precision, f1_score
from utils.preprocessing import binary_partition_by_class, get_stratified_folds, PCA


def main():

    train_data, train_labels, test_data, test_labels = load_data()

    multinomial_logistic_regression(train_data, train_labels, test_data, test_labels)
    # logistic_regression(train_data, train_labels, test_data, test_labels)
    # naive_bayes(train_data, train_labels, test_data, test_labels)
    # nearest_neighbour(train_data, train_labels, test_data, test_labels)
    # neural_net(train_data, train_labels, test_data, test_labels)
    # linear_svm(train_data, train_labels, test_data, test_labels)
    
    # Generate stratified folds
    n_folds = 10
    fold_data, fold_labels = get_stratified_folds(train_data, train_labels, n=n_folds)

    for i in range(n_folds):
        fold_train_data = np.concatenate([fold_data[:i], fold_data[i + 1:]])
        fold_train_data = np.concatenate(fold_train_data)

        fold_train_labels = np.concatenate([fold_labels[:i], fold_labels[i + 1:]])
        fold_train_labels = np.concatenate(fold_train_labels)

        fold_test_data = fold_data[i]

        fold_test_labels = fold_labels[i]


def linear_svm(train_data, train_labels, test_data, test_labels):
    print(f'{LinearSVM.__name__}:')

    # Create and train model
    lsvm_model = LinearSVM(alpha=0.01, features=180)
    model = OneVersusRest(lsvm_model)
    
    model.train(train_data, train_labels)

    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_pred = model.predict(test_data_2k)

    # Print metrics
    print('\nTest Accuracy: {:.02f}%\n'.format(100*accuracy(test_pred, test_labels)))
    mat, classes = confusion_matrix(test_pred, test_labels)
    print('Precision:\n{}\n'.format(np.round(precision(test_pred, test_labels), 2)))
    print('Recall:\n{}\n'.format(np.round(recall(test_pred, test_labels), 2)))
    print('F1:\n{}\n'.format(np.round(f1_score(test_pred, test_labels), 2)))
    print('Confusion Matrix:')
    print(mat)

    # Predict 10000 test set samples and save predictions
    print('Predicting 10k samples...')
    test_pred = model.predict(test_data)
    save_predictions(linear_svm.__name__, test_pred)
    print('Saved 10k predictions.\n')


def logistic_regression(train_data, train_labels, test_data, test_labels):

    print(f'{LogisticRegression.__name__}:')

    # Create and train model
    lr_model = LogisticRegression(train_data.shape[1], eta=0.001, epochs=50)
    model = OneVersusRest(lr_model)

    model.train(train_data, train_labels)

    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_pred = model.predict(test_data_2k)

    # Print metrics
    print('\nTest Accuracy: {:.02f}%\n'.format(100*accuracy(test_pred, test_labels)))
    mat, classes = confusion_matrix(test_pred, test_labels)
    print('Precision:\n{}\n'.format(np.round(precision(test_pred, test_labels), 2)))
    print('Recall:\n{}\n'.format(np.round(recall(test_pred, test_labels), 2)))
    print('F1:\n{}\n'.format(np.round(f1_score(test_pred, test_labels), 2)))
    print('Confusion Matrix:')
    print(mat)

    # Predict 10000 test set samples and save predictions
    print('Predicting 10k samples...')
    test_pred = model.predict(test_data)
    save_predictions(logistic_regression.__name__, test_pred)
    print('Saved 10k predictions.\n')


def multinomial_logistic_regression(train_data, train_labels, test_data, test_labels):

    print(f'{MultinomialLogisticRegression.__name__}:')

    # Create and train model
    model = MultinomialLogisticRegression(
        train_data.shape[1],
        len(np.unique(train_labels)),
        eta=0.001,
        epochs=5 #250
    )

    model.train(train_data, train_labels)

    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_pred, test_probs = model.predict(test_data_2k)

    # Print metrics
    print('\nTest Accuracy: {:.02f}%\n'.format(100*accuracy(test_pred, test_labels)))
    mat, classes = confusion_matrix(test_pred, test_labels)
    print('Precision:\n{}\n'.format(np.round(precision(test_pred, test_labels), 2)))
    print('Recall:\n{}\n'.format(np.round(recall(test_pred, test_labels), 2)))
    print('F1:\n{}\n'.format(np.round(f1_score(test_pred, test_labels), 2)))
    print('Confusion Matrix:')
    print(mat)

    # Predict 10000 test set samples and save predictions
    print('Predicting 10k samples...')
    test_pred, test_probs = model.predict(test_data)
    save_predictions(multinomial_logistic_regression.__name__, test_pred)
    print('Saved 10k predictions.\n')


def naive_bayes(train_data, train_labels, test_data, test_labels):

    print(f'{MultinomialNaiveBayes.__name__}:')

    # Create and train model
    model = MultinomialNaiveBayes()
    model.train(train_data, train_labels)
    
    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_pred = model.predict(test_data_2k)

    # Print metrics
    print('\nTest Accuracy: {:.02f}%\n'.format(100*accuracy(test_pred, test_labels)))
    mat, classes = confusion_matrix(test_pred, test_labels)
    print('Precision:\n{}\n'.format(np.round(precision(test_pred, test_labels), 2)))
    print('Recall:\n{}\n'.format(np.round(recall(test_pred, test_labels), 2)))
    print('F1:\n{}\n'.format(np.round(f1_score(test_pred, test_labels), 2)))
    print('Confusion Matrix:')
    print(mat)

    # Predict 10000 test set samples and save predictions
    print('Predicting 10k samples...')
    test_pred = model.predict(test_data)
    save_predictions(naive_bayes.__name__, test_pred)
    print('Saved 10k predictions.\n')


def nearest_neighbour(train_data, train_labels, test_data, test_labels):

    print(f'{NearestNeighbour.__name__}:')

    # Create and train model
    model = NearestNeighbour(5, dist=manhattan)
    model.train(train_data, train_labels)
    
    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_pred = model.predict(test_data_2k)

    # Print metrics
    print('\nTest Accuracy: {:.02f}%\n'.format(100*accuracy(test_pred, test_labels)))
    mat, classes = confusion_matrix(test_pred, test_labels)
    print('Precision:\n{}\n'.format(np.round(precision(test_pred, test_labels), 2)))
    print('Recall:\n{}\n'.format(np.round(recall(test_pred, test_labels), 2)))
    print('F1:\n{}\n'.format(np.round(f1_score(test_pred, test_labels), 2)))
    print('Confusion Matrix:')
    print(mat)

    # Predict 10000 test set samples and save predictions
    print('Predicting 10k samples...')
    test_pred = model.predict(test_data)
    save_predictions(nearest_neighbour.__name__, test_pred)
    print('Saved 10k predictions.\n')

def neural_net(train_data, train_labels, test_data, test_labels):

    print(f'{NeuralNetwork.__name__}:')

    # Create and train model
    model = NeuralNetwork([
        FlatDenseLayer((784,), activation=tanh),
        FlatDenseLayer((100,), activation=tanh),
        FlatDenseLayer((20,), activation=tanh),
        FlatDenseLayer((10,), activation=sigmoid),
    ], eta=0.01, batch_size=64, epochs=250)

    model.train(train_data, train_labels)

    # Predict 2000 validation set samples and calculate accuracy
    test_data_2k = test_data[:len(test_labels)]
    test_activations, test_pred = model.predict(test_data_2k)
    
    # Print metrics
    print('\nTest Accuracy: {:.02f}%\n'.format(100*accuracy(test_pred, test_labels)))
    mat, classes = confusion_matrix(test_pred, test_labels)
    print('Precision:\n{}\n'.format(np.round(precision(test_pred, test_labels), 2)))
    print('Recall:\n{}\n'.format(np.round(recall(test_pred, test_labels), 2)))
    print('F1:\n{}\n'.format(np.round(f1_score(test_pred, test_labels), 2)))
    print('Confusion Matrix:')
    print(mat)

    # Predict 10000 test set samples and save predictions
    print('Predicting 10k samples...')
    test_activations, test_pred = model.predict(test_data)
    print(len(test_pred))
    save_predictions(neural_net.__name__, test_pred)
    print('Saved 10k predictions.\n')


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
