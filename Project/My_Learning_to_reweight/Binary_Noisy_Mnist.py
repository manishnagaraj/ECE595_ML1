#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:00:56 2021

@author: mnagara
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
        'noisy_prop':0.1,
        'noisy_prop_val':0.0,
    }
    
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_noise', proportion=[0.995], val_proportion=[0.5])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Binary classifier with uniform 10% noise
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    
    
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_unbiased = ReweightingMachine.acc_log

    plt.figure(figsize=(12,8))
    plt.plot(acc_log_unbiased[:,0],acc_log_unbiased[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()