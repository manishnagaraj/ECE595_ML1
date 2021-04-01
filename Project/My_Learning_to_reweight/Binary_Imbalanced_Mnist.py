#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:00:19 2021

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
        'n_val': 5,
    }
    
    data_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.995])
    hyperparameters['train_test'] = 'test'
    test_loader = Utils.get_mnist_loader(hyperparameters, mode='binary_imb', proportion=[0.5])
    
    # Binary classifier
    net = Utils.LeNetBinary(n_out=1).to(device)
    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    

    plot_step = 100

    accuracy_log = []
    
    val_data = data_loader.dataset.data_val.to(device)
    val_labels = data_loader.dataset.labels_val.to(device)
    
    for i in tqdm(range(hyperparameters['num_iterations'])):
        net.train()
        # Line 2 get batch of data
        image, labels = next(iter(data_loader))
        image, labels = image.to(device), labels.to(device)
        # since validation data is small I just fixed them instead of building an iterator
        # initialize a dummy network for the meta learning of the weights
        meta_net = Utils.LeNetBinary(n_out=1)
        meta_net.load_state_dict(net.state_dict())
        meta_net = meta_net.to(device)
        
        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat  = meta_net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat,labels, reduce=False)
        eps = Variable(torch.zeros(cost.size()).to(device), requires_grad=True)
        l_f_meta = torch.sum(cost * eps)
        
        meta_net.zero_grad()        
        
        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(hyperparameters['lr'], source_params=grads)
        
        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)

        l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat,val_labels)

        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
        
        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps,min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        y_f_hat = net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
        l_f = torch.sum(cost * w)

        opt.zero_grad()
        l_f.backward()
        opt.step()

        
        if i % plot_step == 0:
            net.eval()

            acc = []
            for itr,(test_img, test_label) in enumerate(test_loader):
                test_img = test_img.to(device)
                test_label = test_label.to(device)

                output = net(test_img)
                predicted = (F.sigmoid(output) > 0.5).int()

                acc.append((predicted.int() == test_label.int()).float())

            accuracy = torch.cat(acc,dim=0).mean()
            accuracy_log.append(np.array([i,accuracy])[None])



    plt.figure(figsize=(12,8))
    acc_log = np.concatenate(accuracy_log, axis=0)
    plt.plot(acc_log[:,0],acc_log[:,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()