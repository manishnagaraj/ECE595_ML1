#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:14:34 2021

@author: mnagara

binary imbalanced varying proportions
"""

# Import libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import IPython
import gc
import matplotlib
import numpy as np

# Import support files
import Utils

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark=True

if __name__=='__main__':
    
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 100,
        'num_iterations' : 8000,
        'classes' : [9,4],
        'train_test' : 'train',
        'n_items': 5000,
        'n_val': 10,
    }
    
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.995], val_proportion=[0.5])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Binary classifier
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    
    # Biased
    hyperparameters['train_test'] = 'train'
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.9], val_proportion=[0.2])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_1 = ReweightingMachine.acc_log
    
    # Biased
    hyperparameters['train_test'] = 'train'
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.8], val_proportion=[0.2])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_2 = ReweightingMachine.acc_log
    
    # Biased
    hyperparameters['train_test'] = 'train'
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.75], val_proportion=[0.2])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_3 = ReweightingMachine.acc_log
    
    # Biased
    hyperparameters['train_test'] = 'train'
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5], val_proportion=[0.2])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_4 = ReweightingMachine.acc_log
    
    plt.figure(figsize=(12,8))
    plt.plot(acc_log_1[:,0],acc_log_1[:,1],'b')
    plt.plot(acc_log_2[:,0],acc_log_2[:,1],'r')
    plt.plot(acc_log_3[:,0],acc_log_3[:,1],'g')
    plt.plot(acc_log_4[:,0],acc_log_4[:,1],'m')
    plt.legend(['0.9 Proporion', '0.8 Proporion', '0.75 Proporion', '0.5 Proporion'])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()
    