import numpy as np
from scipy.linalg import svd


def get_stratified_folds(x: np.ndarray, y: np.ndarray, n: int=10):
    y = y[:, np.newaxis]
    data = np.concatenate((x, y), axis=1)
    classes = np.unique(y)
    class_split = []
    folds = [[] for i in range(n)]

    leftover = []

    for c in classes:
        # Split dataset by class and get n folds for each class
        c_data = data[data[:, -1] == c]
        split_size = int(c_data.shape[0] / n)
        split_idxs = list(range(split_size, c_data.shape[0], split_size))
        c_data_split = np.array_split(c_data, split_idxs)

        # Take out any samples that are leftover.
        # These will be added in at the end to ensure
        # proper stratification of folds.
        remainder  = c_data.shape[0] % n
        if remainder != 0:
            leftover_split = c_data_split.pop()
            leftover.append(leftover_split)

        # Merge n folds for each class into n folds containing all classes
        class_split.append(c_data_split)
        fold_idx = 0
        for split in c_data_split:
            folds[fold_idx].append(split)
            fold_idx += 1

    # Turn folds to numpy arrays
    folds_np = []
    for fold in folds:
        fold_np = np.concatenate(fold)
        folds_np.append(fold_np)

    # Distribute leftovers across folds.
    leftover = np.concatenate(leftover)
    for i in range(leftover.shape[0]):
        fold_idx = i % n
        leftover_split = leftover[i][np.newaxis, :]
        folds_np[fold_idx] = np.concatenate([folds_np[fold_idx], leftover_split])

    # Prove folds are stratified.
    # for fold in folds_np:
    #     print(fold.shape)
    #     unique, counts = np.unique(fold[:, -1], return_counts=True)
    #     print(tuple(zip(unique, counts)))
    #     print()

    folds_np = np.stack(folds_np)
    return folds_np[:, :, :-1], folds_np[:, :, -1]


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