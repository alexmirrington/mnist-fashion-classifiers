import numpy as np

# Distance functions
def manhattan(x: np.ndarray, y: np.ndarray, axis: int):
    return np.sum(np.abs(x - y), axis=axis)

def euclidean(x: np.ndarray, y: np.ndarray, axis: int):
    return minkowski(x, y, 2, axis=axis)

def euclidean_squared(x: np.ndarray, y: np.ndarray, axis: int):
    return np.sum(np.abs(x - y)**2, axis=axis)

def minkowski(x: np.ndarray, y: np.ndarray, pow: int, axis: int):
    return np.sum(np.abs(x - y)**pow, axis=axis)**(1/pow)

# Activation functions
class ActivationFunction:

    def __init__(self, func: callable, d_func: callable):
        self.func = func
        self.d_func = d_func

def f_sigmoid(x: np.array):
    return 1 / (1 + np.exp(-x))

def df_sigmoid(x: np.array):
    sig_x = f_sigmoid(x)
    return sig_x * (1 - sig_x)

def f_tanh(x: np.array):
    e_nx = np.exp(-x)
    e_x = np.exp(x)
    return (e_x - e_nx)/(e_x + e_nx)

def df_tanh(x: np.array):
    tanh_x = f_tanh(x)
    return 1 - tanh_x**2

def f_relu(x: np.array):
    x_copy = np.copy(x)
    x_copy[x_copy<0] = 0
    return x_copy

def df_relu(x: np.array):
    ret = (x > 0).astype(int)
    return ret

def f_softplus(x: np.array):
    e_x = np.exp(x)
    return np.log(1 + e_x)

def df_softplus(x: np.array):
    return f_sigmoid(x)


sigmoid = ActivationFunction(f_sigmoid, df_sigmoid)
tanh = ActivationFunction(f_tanh, df_tanh)
relu = ActivationFunction(f_relu, df_relu)
softplus = ActivationFunction(f_softplus, df_softplus)

# Error functions
class ErrorFunction:

    def __init__(self, func: callable, d_func: callable):
        self.func = func
        self.d_func = d_func

def f_sse(predicted: np.ndarray, actual: np.ndarray):
    return 0.5*(actual - predicted)**2

def df_sse(predicted: np.ndarray, actual: np.ndarray):
    return actual - predicted

sse = ErrorFunction(f_sse, df_sse)
