import pandas as pd
import numpy as np
import h5py
from .classifier import Classifier


class MultinomialNaiveBayes(Classifier):

    def __init__(self, alpha=1.0):
        # Additive (Laplace/Lidstone) smoothing parameter,
        # used to avoid zero theta_ck probabilities. Works
        # better for larger training sets.
        self.alpha = alpha

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Compute prior (pi_k) probability for the classes in y,
        and posterior (theta_ck) probabilities for attributes given classes.

        These are used to predict the class of new samples.
        """
        # Separate into classes
        self.classes = np.unique(y)        
        data_by_class = [[data for data, lbl in zip(x, y) if lbl == c] for c in self.classes]
        
        # Compute prior probabilities
        data_count = x.shape[0]
        self.pi_c = np.array([len(class_data) / data_count for class_data in data_by_class])
        
        # Compute posterior probabilities
        attrib_counts_by_class = np.array([np.sum(class_data, axis=0) for class_data in data_by_class]) + self.alpha
        total_counts_by_class = np.sum(attrib_counts_by_class, axis=1)
        self.theta_ck = attrib_counts_by_class / total_counts_by_class[:, np.newaxis]

    def get_log_likelihood(self, x):
        """
        Get the log-likelihood for a new set of data x.
        """

        log_pi_c = np.log(self.pi_c)
        log_theta_ck = np.log(self.theta_ck)

        # Sum up x_k theta_k for each class for each sample,
        # then add log_pi_c for each sample.
        return (log_theta_ck @ x.T + log_pi_c[:, np.newaxis]).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.get_log_likelihood(x), axis=1)
        

