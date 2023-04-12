import time

import numpy as np


class LROptimizer(object):
    """
    Logistic regression optimizer using coordinate descent (CD) and gradient descent (GD)

    Arguments:
    W -- weights, a numpy array of size (n_x, 1)
    lr -- learning rate of the gradient descent update rule
    block -- block of coordinates to update if
    """

    def __init__(self, W, lr=0.01, block=None):
        self.W = W
        self.lr = lr
        self.block = block
        self.grads = []

    def optimize(self, y, y_pred, X):
        if self.block is None:
            grad, gtime = self.gradient_descent(y, y_pred, X)

        else:
            subX = X[:, self.block]
            grad, gtime = self.coordinate_descent(y, y_pred, subX)
        return grad, gtime

    @staticmethod
    def loss(y, y_pred):
        return -(1.0 / len(y)) * np.sum(y * np.log(y_pred) + (1.0 - y) * np.log(1.0 - y_pred + 1e-7))

    @staticmethod
    def gradient_descent(y, y_pred, X):
        t = time.time()
        m = X.shape[0]
        dw = 1 / m * X.T @ (y_pred - y)
        gtime = time.time() - t
        return dw, gtime

    @staticmethod
    def coordinate_descent(y, y_pred, X):
        t = time.time()
        m = X.shape[0]
        dw = 1 / m * X.T @ (y_pred - y)
        gtime = time.time() - t
        return dw, gtime
