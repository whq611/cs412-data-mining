import numpy as np

class Logistic:
    def __init__(self, lr):
        self.w = None
        self.lr = lr

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w)))

    def update(self, X, h, y):
    	# update the weight using gradient ascent
        self.w += self.lr * np.dot(X.T, y.reshape(np.shape(y)[0], 1) - h)

    def likelihood(self, X, y):
    	# calculate the log likelihood
        return np.sum(y*np.dot(X, self.w) - np.log(1 + np.exp(np.dot(X, self.w))))