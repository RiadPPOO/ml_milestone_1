import numpy as np
from src.utils import append_bias_term

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.w = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        training_data_bias = append_bias_term(training_data)
        one_hot_labels = label_to_onehot(training_labels)
        self.w = np.zeros((len(training_data_bias[0]), len(one_hot_labels[0])))

        for i in range (self.max_iters):
            self.w = self.w - self.lr * training_data_bias.T @ (self.softmax(training_data_bias @ self.w) - one_hot_labels)

        pred_labels = onehot_to_label(self.softmax(training_data_bias@self.w))
        return pred_labels
    
    def softmax (self, Z):
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_stable)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        test_data_bias = append_bias_term(test_data)
        pred_labels = onehot_to_label(self.softmax(test_data_bias@self.w))
        return pred_labels
