import numpy as np

def shuffle_data(x: np.ndarray, y:  np.ndarray):
    data = np.concatenate((x, y), axis=1)
    np.random.shuffle(data)
    return data[:,:-1], data[:,data.shape[1]-1:]