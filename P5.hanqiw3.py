import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm
import time
from dataloader import DataLoader
from LR import Logistic


def plot_acc(acc, epochs):
    indices = [i for i in range(1, epochs+1)]
    plt.plot(indices, acc)
    plt.title("test accuracy vs. the number of iterations")
    plt.ylabel("test accuracy")
    plt.xlabel("number of iterations")
    plt.show()

def plot_ll(ll, epochs):
    indices = [i for i in range(1, epochs+1)]
    ll = [-i for i in ll]
    plt.plot(indices, ll)
    plt.title("training loss vs. the number of iterations")
    plt.ylabel("training loss")
    plt.xlabel("number of iterations")
    plt.show()

def train_LR(Logistic, epochs, Xtrn, Ytrn, Xtst, Ytst):
    tst_acc = []
    log_likelihood = []
    Logistic.w = np.random.normal(0, .05, size=(np.shape(Xtrn)[1], 1))

    for _ in range(epochs):
        h = logistic.sigmoid(Xtrn)
        logistic.update(Xtrn, h, Ytrn) #update weights

        ll = logistic.likelihood(Xtrn, Ytrn) # likelihood
        log_likelihood.append(ll)

        pred = np.array([1 if h >= 0.5 else 0 for h in logistic.sigmoid(Xtst)]) # prediction
        acc = (pred == Ytst).sum() / len(Ytst)
        tst_acc.append(acc)

    plot_acc(tst_acc, epochs) #plot accuracy
    plot_ll(log_likelihood, epochs) # plot likelihood
if __name__ == '__main__':
    # preprocess data
    dataloader = DataLoader(Xtrainpath='data/train_data.mat',
                            Ytrainpath='data/train_data.mat',
                            Xtestpath='data/test_data.mat',
                            Ytestpath='data/test_data.mat')
    Xtrain, Ytrain, Xtest, Ytest = dataloader.load_data()
    Xtrn = np.copy(Xtrain)
    Xtst = np.copy(Xtest)
    Xtrn = np.reshape(Xtrn, (np.shape(Xtrn)[0], -1)) # flatten
    Xtst = np.reshape(Xtst, (np.shape(Xtst)[0], -1))
    Xtrn = np.true_divide(Xtrn, 255) # normalize
    Xtst = np.true_divide(Xtst, 255)
    Ytrn = np.array([1 if y == 1 else 0 for y in Ytrain]) # convert into binary
    Ytst = np.array([1 if y == 1 else 0 for y in Ytest])
    Ytrn = Ytrn.reshape(np.shape(Ytrn)[0], 1)

    LR = 0.1
    Epochs = 100
    logistic = Logistic(LR)

    train_LR(logistic, Epochs, Xtrn, Ytrn, Xtst, Ytst)