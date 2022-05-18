from scipy.io import loadmat
import gzip
import numpy as np


class DataLoader(object):
    """
    class to load MNIST data
    """
    def __init__(self, Xtrainpath, Ytrainpath, Xtestpath, Ytestpath):
        self.Xtrainpath = Xtrainpath
        self.Ytrainpath = Ytrainpath
        self.Xtestpath = Xtestpath
        self.Ytestpath = Ytestpath

    @staticmethod
    def get_images(filename):
        mat = loadmat(filename)
        return mat['X']

    @staticmethod
    def get_labels(filename):
        mat = loadmat(filename)
        return mat['Y']

    def load_data(self):
        Xtrain = self.get_images(self.Xtrainpath)
        Ytrain = self.get_labels(self.Ytrainpath)
        Xtest = self.get_images(self.Xtestpath)
        Ytest = self.get_labels(self.Xtestpath)
        return Xtrain, Ytrain, Xtest, Ytest

