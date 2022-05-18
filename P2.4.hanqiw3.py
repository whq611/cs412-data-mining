from xml.etree.ElementPath import prepare_descendant
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm
import time
from dataloader import DataLoader
from MLP import MLP
from tqdm import tqdm

torch.manual_seed(42)
class MyDataSet(Dataset):
    def __init__(self, X, y):
        self.data = torch.FloatTensor(X)
        self.label = torch.LongTensor(y)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        return x, y
  
    def __len__(self):
        return len(self.data)
def accuracy(output, labels):
    preds = output.argmax(dim=1)
    correct = (preds == labels).sum().float() # count correct num of pred
    acc = correct / len(labels)
    return acc

def plot_loss(trn, tst, trn_std, tst_std, epochs):
    indices = [i for i in range(1, epochs+1)]
    plt.errorbar(indices, trn, yerr=trn_std, label="train")
    plt.errorbar(indices, tst, yerr=tst_std, label="test")
    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    
def plot_acc(trn, tst, trn_std, tst_std, epochs):
    indices = [i for i in range(1, epochs+1)]
    plt.errorbar(indices, trn, yerr=trn_std, label="train")
    plt.errorbar(indices, tst, yerr=tst_std, label="test")
    plt.title("Accuracy Plot")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
def train(epochs, n_iter, trn_data, tst_data, device):
    trn_loss_dict, tst_loss_dict = np.zeros((n_iter, epochs)), np.zeros((n_iter, epochs))
    trn_acc_dict, tst_acc_dict = np.zeros((n_iter, epochs)), np.zeros((n_iter, epochs))
    
    for i in range(n_iter):
        trn_loss_list, tst_loss_list = np.array([]), np.array([])
        trn_acc_list, tst_acc_list = np.array([]), np.array([])
        model = MLP(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-6)
        criterion = nn.CrossEntropyLoss().to(device)

        for _ in tqdm(range(epochs)):
            start = time.time()
            trn_loss, tst_loss = 0, 0
            trn_acc, tst_acc = 0, 0

            # training(w/ gradient descent)
            model.train()
            for data, label in trn_data:
                data, label = data.to(device), label.to(device)
                label = label.squeeze(1).long()

                optimizer.zero_grad()
                output = model(data)
                acc = accuracy(output, label)
                #print("train: ",acc)
                
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                


                trn_loss += loss.item()
                trn_acc += acc.item()
            trn_loss_list = np.append(trn_loss_list, trn_loss / len(trn_data))
            trn_acc_list = np.append(trn_acc_list, trn_acc / len(trn_data))
            
            # testing(w/ no gradient descent)
            model.eval()
            for data, label in tst_data:
                data, label = data.to(device), label.to(device)
                label = label.squeeze(1).long()

                with torch.no_grad():
                    output = model(data)
                    acc = accuracy(output, label)
                    #print("eval: ",acc)

                    
                    loss = criterion(output, label)
                    tst_acc += acc.item()
                    tst_loss += loss.item()
            tst_loss_list = np.append(tst_loss_list, tst_loss / len(tst_data))
            tst_acc_list = np.append(tst_acc_list, tst_acc / len(tst_data))
        
        trn_loss_dict[i] = np.add(trn_loss_dict[i], trn_loss_list)
        trn_acc_dict[i] = np.add(trn_acc_dict[i], trn_acc_list)
        tst_loss_dict[i] = np.add(tst_loss_dict[i], tst_loss_list)
        tst_acc_dict[i] = np.add(tst_acc_dict[i], tst_acc_list)

    plot_loss(np.mean(trn_loss_dict, axis=0), np.mean(tst_loss_dict, axis=0), np.std(trn_loss_dict, axis=0), np.std(tst_loss_dict, axis=0), epochs) #plot loss
    plot_acc(np.mean(trn_acc_dict, axis=0), np.mean(tst_acc_dict, axis=0), np.std(trn_acc_dict, axis=0), np.std(tst_acc_dict, axis=0), epochs) # plot 


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # preprocess data
    LR = 0.001
    BATCH_SIZE = 256
    N_EPOCHS = 40
    N_ITER = 5
    NUM_CLASSES = 10
    INPUT_DIM = 784
    HIDDEN_DIM = 32
    dataloader = DataLoader(Xtrainpath='data/train_data.mat',
                            Ytrainpath='data/train_data.mat',
                            Xtestpath='data/test_data.mat',
                            Ytestpath='data/test_data.mat')
    Xtrain, Ytrain, Xtest, Ytest = dataloader.load_data()
    # load data
    trn = MyDataSet(Xtrain, Ytrain)
    tst = MyDataSet(Xtest, Ytest)
    trn = data.DataLoader(trn, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    tst = data.DataLoader(tst, batch_size=1000, drop_last=True, shuffle=False)
    
    # load model
    
    train(N_EPOCHS, N_ITER, trn, tst, device)
    #train(mlp, N_EPOCHS, N_ITER, trn, tst, optimizer, criterion, device)