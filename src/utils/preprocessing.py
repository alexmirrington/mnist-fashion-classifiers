import numpy as np
from scipy.linalg import svd


def shuffle_data(x: np.ndarray, y:  np.ndarray):
    y = y[:, np.newaxis]
    data = np.concatenate((x, y), axis=1)
    np.random.shuffle(data)
    return data[:,:-1], data[:,data.shape[1]-1:].squeeze()

def binary_partition_by_class(x: np.ndarray, y: np.ndarray, partition_map: set):
    classes = np.unique(y)
    for c in partition_map:
        if c not in classes:
            raise ValueError('Value {} in parition_map was not found in y.'.format(c))

    partition = np.copy(y)
    for c in partition_map:
        partition = (partition==c).astype('uint8')

    return x, partition

# @returns: projected vector, explained variance for all components
# PCA based on SVD in order to save computational resources and have more accurate results#
# Reference (with links to papers) https://stats.stackexchange.com/questions/79043/why-pca-of-data-by-means-of-svd-of-the-data
def PCA(A: np.ndarray, n_components=6):
    E = np.mean(A, axis=0)
    C = A - E
    u, s, v = svd(C)

    explained_var = np.cumsum(s) / np.sum(s)
    proj_A = np.dot(C,v[:,range(0,n_components)])

    return proj_A, explained_var