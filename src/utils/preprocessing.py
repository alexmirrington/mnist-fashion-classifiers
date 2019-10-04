import numpy as np

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

    partition = y
    for c in partition_map:
        partition[partition==c] = 0
        partition[partition!=c] = 1

    return x, partition