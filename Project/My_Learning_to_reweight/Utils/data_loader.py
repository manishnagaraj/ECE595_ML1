#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:58:42 2021

@author: mnagara

Data Loader 

(1) Class Imbalance
(2) Noise --- 
    (a) Uniform Noise
    (b) Targetted Noise
    (c) Background Noise?
"""
import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image
import sys
# import h5py
import numpy as np
import collections
import numbers
import math
import pandas as pd


############################################################################################
############################################################################################

class MNISTImbalanced_Binary():
    def __init__(self, n_items = 5000, classes=[9, 4], proportion=0.9, n_val=5, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('/home/min/a/mnagara/Desktop/Pytorch_experiments/Data', train=True, download=True)
        else:
            self.mnist = datasets.MNIST('/home/min/a/mnagara/Desktop/Pytorch_experiments/Data', train=False, download=True)
            n_val = 0
            
        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        n_class = [0, 0]
        n_class[0] = int(np.floor(n_items*proportion))
        n_class[1] = n_items - n_class[0]

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        if mode == "train":
            data_source = self.mnist.train_data
            label_source = self.mnist.train_labels
        else:
            data_source = self.mnist.test_data
            label_source = self.mnist.test_labels

        for i, c in enumerate(classes):
            tmp_idx = np.where(label_source == c)[0]
            np.random.shuffle(tmp_idx)
            tmp_idx = torch.from_numpy(tmp_idx)
            img = data_source[tmp_idx[:n_class[i] - n_val]]
            self.data.append(img)
            
            cl = label_source[tmp_idx[:n_class[i] - n_val]]
            self.labels.append((cl == classes[0]).float())

            if mode == "train":
                img_val = data_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                for idx in range(img_val.size(0)):
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)

                    self.data_val.append(img_tmp.unsqueeze(0))

                cl_val = label_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                self.labels_val.append((cl_val == classes[0]).float())

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        if mode == "train":
            self.data_val = torch.cat(self.data_val, dim=0)
            self.labels_val = torch.cat(self.labels_val, dim=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    
################################################################################################

class MNISTImbalanced():
    def __init__(self, n_items = 50000, proportion=0.1*np.ones(10), n_val=2, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('/home/min/a/mnagara/Desktop/Pytorch_experiments/Data', train=True, download=True)
        else:
            self.mnist = datasets.MNIST('/home/min/a/mnagara/Desktop/Pytorch_experiments/Data', train=False, download=True)

        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        n_class = np.zeros(10)
        classes = np.arange(10)
        assert proportion.sum()==1, 'Check Proportions'
        
        for i in range(10):
            n_class[i] = int(np.floor(n_items*proportion[i]))
        # n_class[1] = n_items - n_class[0]

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        if mode == "train":
            data_source = self.mnist.train_data
            label_source = self.mnist.train_labels
        else:
            data_source = self.mnist.test_data
            label_source = self.mnist.test_labels

        for i, c in enumerate(classes):
            tmp_idx = np.where(label_source == c)[0]
            np.random.shuffle(tmp_idx)
            tmp_idx = torch.from_numpy(tmp_idx)
            img = data_source[tmp_idx[:n_class[i] - n_val]]
            self.data.append(img)
            
            cl = label_source[tmp_idx[:n_class[i] - n_val]]
            self.labels.append(cl.float())

            if mode == "train":
                img_val = data_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                for idx in range(img_val.size(0)):
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)

                    self.data_val.append(img_tmp.unsqueeze(0))

                cl_val = label_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                self.labels_val.append((cl_val == classes[0]).float())

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        if mode == "train":
            self.data_val = torch.cat(self.data_val, dim=0)
            self.labels_val = torch.cat(self.labels_val, dim=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
################################################################################################

class MNISTUniformNoisy_Binary():
    def __init__(self, n_items = 5000, classes=[9, 4], noisy_prop=0.0, noisy_prop_val=0.1, n_val=5, random_seed=1, mode="train"):
        if mode == "train":
            self.mnist = datasets.MNIST('/home/min/a/mnagara/Desktop/Pytorch_experiments/Data', train=True, download=True)
        else:
            self.mnist = datasets.MNIST('/home/min/a/mnagara/Desktop/Pytorch_experiments/Data', train=False, download=True)

        self.transform=transforms.Compose([
               transforms.Resize([32,32]),
               transforms.ToTensor(),
               transforms.Normalize((0.1307,), (0.3081,))
            ])
        n_class = [0, 0]
        n_class[0] = int(np.floor(n_items*0.5))
        n_class[1] = n_items - n_class[0]

        self.data = []
        self.data_val = []
        self.labels = []
        self.labels_val = []

        if mode == "train":
            data_source = self.mnist.train_data
            label_source = self.mnist.train_labels
        else:
            data_source = self.mnist.test_data
            label_source = self.mnist.test_labels

        for i, c in enumerate(classes):
            other_labels= np.arange(10)
            np.delete(other_labels,[c])
            tmp_idx = np.where(label_source == c)[0]
            np.random.shuffle(tmp_idx)
            tmp_idx = torch.from_numpy(tmp_idx)
            img = data_source[tmp_idx[:n_class[i] - n_val]]
            self.data.append(img)
            
            cl = label_source[tmp_idx[:n_class[i] - n_val]]
            noisy_idx = random.sample(range(0,n_class[i] - n_val-1),noisy_prop*(n_class[i] - n_val))
            for noise in noisy_idx:
                cl[noise] = random.choice(other_labels)
            self.labels.append((cl == classes[0]).float())

            if mode == "train":
                img_val = data_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                for idx in range(img_val.size(0)):
                    img_tmp = Image.fromarray(img_val[idx].numpy(), mode='L')
                    img_tmp = self.transform(img_tmp)

                    self.data_val.append(img_tmp.unsqueeze(0))

                cl_val = label_source[tmp_idx[n_class[i] - n_val:n_class[i]]]
                if noisy_prop_val > 0:
                    noisy_idx = random.sample(range(0, n_val-1), noisy_prop_val*(n_val))
                    for noise in noisy_idx:
                        cl_val[noise] = random.choice(other_labels)
                self.labels_val.append((cl_val == classes[0]).float())

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

        if mode == "train":
            self.data_val = torch.cat(self.data_val, dim=0)
            self.labels_val = torch.cat(self.labels_val, dim=0)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    

    
################################################################################################
################################################################################################
    
def get_mnist_loader(hp, mode, proportion):
    """Build and return data loader."""
    batch_size = hp['batch_size']
    n_items = hp['n_items']
    train_test = hp['train_test']
    n_val = hp['n_val']
    shuffle = False
    
    if mode == 'binary_imb':
         prop = proportion[0]
         classes = hp['classes']
         dataset = MNISTImbalanced_Binary(classes=classes, n_items=n_items, proportion=prop, n_val=n_val,mode=train_test)
         if train_test == 'train':
             shuffle = True
    elif mode == 'multi_imb':
        dataset = MNISTImbalanced(n_items=n_items, proportion=proportion, n_val=n_val, mode=train_test)
        if train_test == 'train':
            shuffle = True
    elif mode == 'binary_noise':
        classes = hp['classes']
        noisy_prop = hp['noisy_prop']
        noisy_prop_val = hp['noisy_prop_val']
        dataset = MNISTUniformNoisy_Binary(n_items=n_items, classes=classes, noisy_prop=noisy_prop, 
                                           noisy_prop_val=noisy_prop_val, n_val=n_val, mode=train_test)
        if train_test == 'train':
            shuffle = True
    else:
         print('ERROR, not implemented')
         sys.exit()
             
             
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader