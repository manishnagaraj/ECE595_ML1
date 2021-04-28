#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:56:54 2021

@author: mnagara

Generate Noisy binary class 

Based on https://github.com/pxiangwu/PLC/tree/master/cifar
"""

import os
import sys
import torch
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import random
import argparse
import copy
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image

import pdb

# Import support files
import Utils

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark=True

def curropt_data(n_items=5000, classes=[9,4], noisy_prop=0.1, n_val=10, tpe=3):    
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 100,
        'num_iterations' : 8000,
        'classes' : classes,
        'train_test' : 'train',
        'n_items': n_items,
        'n_val': n_val,
    }
    
    # Download MNIST clean binary classifier data
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5], val_proportion=[0.5])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    n = len(data_loader.dataset)
    net = Utils.LeNetBinary(n_out=1).to(device)
     
    # Train for n_epochs regularly   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_unbiased = ReweightingMachine.acc_log

    plt.figure(figsize=(12,8))
    plt.plot(acc_log_unbiased[:,0],acc_log_unbiased[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()              
    
    # for each data_loader image find the softmax value
    # Load fresh unbiased dataloader with batch_size = 1
    
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 1,
        'num_iterations' : 8000,
        'classes' : classes,
        'train_test' : 'train',
        'n_items': n_items,
        'n_val': n_val,
    }
    
    # Download MNIST clean binary classifier data
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5], val_proportion=[0.5])
    # see the output
    eta = np.zeros([n,2])
    trl = torch.zeros([n, 2])
    
    # Create curropted data:
    curr_data = []
    curr_labels = []
    data_source = data_loader.dataset
    
    for batch_idx, (image, labels) in enumerate(data_loader):
        # if batch_idx < 5:
        #     plt.figure()
        #     plt.imshow(image[0,0])
        image = image.to(device)
        eta[batch_idx, 0] = (torch.sigmoid(net(image))).cpu().detach().numpy()
        trl[batch_idx, 0] = 1-(torch.sigmoid(net(image))).detach()
        trl[batch_idx, 1] = (torch.sigmoid(net(image))).detach()
        eta[batch_idx, 1] = int(batch_idx)
    
    y_idx_tmp = []
    for j in range(len(trl)):
        y_idx_tmp.append(torch.multinomial(trl[j], 1))
        
    y_train = np.array(y_idx_tmp)
    eta_u = eta[:,0]
    if tpe == 1:
        f_us = 2*eta_u*(eta_u-1/2)**2
    elif tpe == 2:
        f_us = (eta_u >= 1/2)*(1-eta_u) + (eta_u < 1/2)*eta_u
    else:
        f_us = -2*(eta_u-1/2)**2 + 1/2
    count = 0
    
    for i in range(len(y_train)):
        if y_train[i]==1:
            y_train[i] = torch.tensor(np.random.binomial(1, 1 - f_us[i], 1), dtype=torch.long)
            count+= 1
            
    print("\n Corrupted Size {} | Noisy Level {:.3f}% \n".format(count, count/float(len(y_train))*100))
            
    if tpe == 4:
        eta_sort = eta[np.argsort(eta[:, 0])]
        break_pnt = len(eta_sort)
        for i in range(len(eta_sort)):
            if eta_sort[i][0] >= 0.5 :
                break_pnt = i
                break
            
        ### Curropt what noise_prop values
        noise_prop_values = int(noisy_prop* len(eta_sort))  
        # on each side -- 0.5
        noise_prop_value_lower_idx = int(np.floor(break_pnt - 0.5*noise_prop_values))
        noise_prop_value_upper_idx = int(np.ceil(break_pnt + 0.5*noise_prop_values))
        
        lower_idxs=[]
        upper_idxs=[]
        for i in range(len(eta_sort)):
            if (i >= noise_prop_value_lower_idx) and (i < break_pnt):
                lower_idxs.append(int(eta_sort[i,1]))
            if (i <= noise_prop_value_upper_idx) and (i > break_pnt):
                upper_idxs.append(int(eta_sort[i,1]))
        
        for i in range(len(data_source)):
            if (i in lower_idxs) or (i in upper_idxs):
                y_train[i] = 0 if data_source.labels[i] == 1 else 1
            else:
                y_train[i] = data_source.labels[i]           
    
    for i in range(len(data_source)):
        curr_data.append(data_source.data[i])
        curr_labels.append(torch.tensor(y_train[i]).float())

    # pdb.set_trace()
    return curr_data, curr_labels, data_source.data_val, data_source.labels_val


class TgtNoisyMNISTBinary():
    def __init__(self, n_items = 5000, classes=[9, 4], noisy_prop=0.1, n_val=5, random_seed=1, tpe=3):
        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        
        self.data, self.labels, self.data_val, self.labels_val = curropt_data(n_items, classes, noisy_prop, n_val, tpe)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target    
            
    
if __name__ == '__main__':

    
    # Type 1
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 100,
        'num_iterations' : 8000,
        'classes' : [9,4],
        'train_test' : 'train',
        'n_items': 5000,
        'n_val': 10,
        'tpe': 1
    }
    
    # data, labels, data_val, labels_val = curropt_data(n_items=5000, classes=[9,4], noisy_prop=0.1, n_val=10)
    
    dataset_p = TgtNoisyMNISTBinary(n_items=hyperparameters['n_items'], classes=hyperparameters['classes'], noisy_prop=0.35, n_val=hyperparameters['n_val'], tpe=hyperparameters['tpe']) 
    train_loader = DataLoader(dataset=dataset_p, batch_size=hyperparameters['batch_size'], shuffle=True)
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Train with this curropted data
    net = Utils.LeNetBinary(n_out=1).to(device)
    # Train for n_epochs regularly   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(train_loader, test_loader)
    acc_log_noisy_tgt_1 = ReweightingMachine.acc_log

    plt.figure(figsize=(12,8))
    plt.plot(acc_log_noisy_tgt_1[:,0],acc_log_noisy_tgt_1[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()                  
        
    
    # Type 2
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 100,
        'num_iterations' : 8000,
        'classes' : [9,4],
        'train_test' : 'train',
        'n_items': 5000,
        'n_val': 10,
        'tpe': 2
    }
    
    # data, labels, data_val, labels_val = curropt_data(n_items=5000, classes=[9,4], noisy_prop=0.1, n_val=10)
    
    dataset_p = TgtNoisyMNISTBinary(n_items=hyperparameters['n_items'], classes=hyperparameters['classes'], noisy_prop=0.35, n_val=hyperparameters['n_val'], tpe=hyperparameters['tpe']) 
    train_loader = DataLoader(dataset=dataset_p, batch_size=hyperparameters['batch_size'], shuffle=True)
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Train with this curropted data
    net = Utils.LeNetBinary(n_out=1).to(device)
    # Train for n_epochs regularly   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(train_loader, test_loader)
    acc_log_noisy_tgt_2 = ReweightingMachine.acc_log

    plt.figure(figsize=(12,8))
    plt.plot(acc_log_noisy_tgt_1[:,0],acc_log_noisy_tgt_1[:,1])
    plt.plot(acc_log_noisy_tgt_2[:,0],acc_log_noisy_tgt_2[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['Type 1 Noise', 'Type 2 Noise'])
    plt.grid()
    plt.show()  
    
    # Type 3
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 100,
        'num_iterations' : 8000,
        'classes' : [9,4],
        'train_test' : 'train',
        'n_items': 5000,
        'n_val': 10,
        'tpe': 3
    }
    
    # data, labels, data_val, labels_val = curropt_data(n_items=5000, classes=[9,4], noisy_prop=0.1, n_val=10)
    
    dataset_p = TgtNoisyMNISTBinary(n_items=hyperparameters['n_items'], classes=hyperparameters['classes'], noisy_prop=0.35, n_val=hyperparameters['n_val'], tpe=hyperparameters['tpe']) 
    train_loader = DataLoader(dataset=dataset_p, batch_size=hyperparameters['batch_size'], shuffle=True)
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Train with this curropted data
    net = Utils.LeNetBinary(n_out=1).to(device)
    # Train for n_epochs regularly   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(train_loader, test_loader)
    acc_log_noisy_tgt_3 = ReweightingMachine.acc_log

    plt.figure(figsize=(12,8))
    plt.plot(acc_log_noisy_tgt_1[:,0],acc_log_noisy_tgt_1[:,1])
    plt.plot(acc_log_noisy_tgt_2[:,0],acc_log_noisy_tgt_2[:,1])
    plt.plot(acc_log_noisy_tgt_3[:,0],acc_log_noisy_tgt_3[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['Type 1 Noise', 'Type 2 Noise', 'Type 3 Noise'])
    plt.grid()
    plt.show()     
    
    # Type 4
    hyperparameters = {
        'lr' : 1e-3,
        'momentum' : 0.9,
        'batch_size' : 100,
        'num_iterations' : 8000,
        'classes' : [9,4],
        'train_test' : 'train',
        'n_items': 5000,
        'n_val': 10,
        'tpe': 4
    }
    
    # data, labels, data_val, labels_val = curropt_data(n_items=5000, classes=[9,4], noisy_prop=0.1, n_val=10)
    
    dataset_p = TgtNoisyMNISTBinary(n_items=hyperparameters['n_items'], classes=hyperparameters['classes'], noisy_prop=0.35, n_val=hyperparameters['n_val'], tpe=hyperparameters['tpe']) 
    train_loader = DataLoader(dataset=dataset_p, batch_size=hyperparameters['batch_size'], shuffle=True)
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Train with this curropted data
    net = Utils.LeNetBinary(n_out=1).to(device)
    # Train for n_epochs regularly   
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(train_loader, test_loader)
    acc_log_noisy_tgt_4 = ReweightingMachine.acc_log

    plt.figure(figsize=(12,8))
    plt.plot(acc_log_noisy_tgt_1[:,0],acc_log_noisy_tgt_1[:,1])
    plt.plot(acc_log_noisy_tgt_2[:,0],acc_log_noisy_tgt_2[:,1])
    plt.plot(acc_log_noisy_tgt_3[:,0],acc_log_noisy_tgt_3[:,1])
    plt.plot(acc_log_noisy_tgt_4[:,0],acc_log_noisy_tgt_4[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['Type 1 Noise', 'Type 2 Noise', 'Type 3 Noise', 'Type 4 Noise'])
    plt.grid()
    plt.show()     
     
    # Clean data
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
    
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5], val_proportion=[0.5])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Binary classifier
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    
    
    ReweightingMachine = Utils.ReweightingAlgorithm(hyperparameters, net, device)
    ReweightingMachine.learn(data_loader, test_loader)
    acc_log_unbiased = ReweightingMachine.acc_log   
    
    plt.figure(figsize=(12,8))
    plt.plot(acc_log_noisy_tgt_1[:,0],acc_log_noisy_tgt_1[:,1])
    plt.plot(acc_log_noisy_tgt_2[:,0],acc_log_noisy_tgt_2[:,1])
    plt.plot(acc_log_noisy_tgt_3[:,0],acc_log_noisy_tgt_3[:,1])
    plt.plot(acc_log_noisy_tgt_4[:,0],acc_log_noisy_tgt_4[:,1])
    plt.plot(acc_log_unbiased[:,0],acc_log_unbiased[:,1]) 
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend(['Type 1 Noise', 'Type 2 Noise', 'Type 3 Noise', 'Type 4 Noise', 'Clean Data'])
    plt.grid()
    plt.show()    
