import numpy as np

def accuracy(y_hat, y):
    if (y_hat.shape == y.shape):
        pred = (y_hat == y)
        return len(np.where(pred == True)[0]) / len(y)

def confusion_matrix(y_hat, y):
    if (y_hat.shape == y.shape):
        classes = set(np.unique(y_hat)).intersection(np.unique(y))

        mat = [[None for c in classes] for c_hat in classes]
        c_hat_idx = 0

        for c_hat in classes:
            y_hat_equal_c_hat = y_hat == c_hat
            c_idx = 0
            for c in classes:
                y_equal_c = y == c
                mat_entry = np.sum(y_hat_equal_c_hat * y_equal_c)
                # Actual count of class c is sum of column
                # Num samples predicted as class c_hat is sum of row
                mat[c_hat_idx][c_idx] = mat_entry
                c_idx += 1
            c_hat_idx += 1

        mat = np.asarray(mat)
        return mat, classes

def recall(y_hat, y):
    if (y_hat.shape == y.shape):
        conf_mat, classes = confusion_matrix(y_hat, y)
        return np.diagonal(conf_mat) / np.sum(conf_mat, axis=0)

def precision(y_hat, y):
    if (y_hat.shape == y.shape):
        conf_mat, classes = confusion_matrix(y_hat, y)
        return np.diagonal(conf_mat) / np.sum(conf_mat, axis=1)

def f1_score(y_hat, y):
    if (y_hat.shape == y.shape):
        r = recall(y_hat, y)
        p = precision(y_hat, y)
        return (2 * p * r) / (p + r)