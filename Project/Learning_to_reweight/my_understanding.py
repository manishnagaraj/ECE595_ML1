#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 20:04:47 2021

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
import numpy as np
import matplotlib

# Import support files
from data_loader import get_mnist_loader
from model import LeNet

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
    }
    
    data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
    test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")
    
    # Binary classifier
    net = LeNet(n_out=1).to(device)


    opt = torch.optim.SGD(net.params(),lr=hyperparameters["lr"])   
    
    meta_losses_clean = []
    net_losses = []
    plot_step = 100

    smoothing_alpha = 0.9
    
    meta_l = 0
    net_l = 0
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
        meta_net = LeNet(n_out=1)
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

        meta_l = smoothing_alpha *meta_l + (1 - smoothing_alpha)* l_g_meta.item()
        meta_losses_clean.append(meta_l/(1 - smoothing_alpha**(i+1)))

        net_l = smoothing_alpha *net_l + (1 - smoothing_alpha)* l_f.item()
        net_losses.append(net_l/(1 - smoothing_alpha**(i+1)))
        
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


            IPython.display.clear_output()
            fig, axes = plt.subplots(1, 2, figsize=(13,5))
            ax1, ax2 = axes.ravel()

            ax1.plot(meta_losses_clean, label='meta_losses_clean')
            ax1.plot(net_losses, label='net_losses')
            ax1.set_ylabel("Losses")
            ax1.set_xlabel("Iteration")
            ax1.legend()

            acc_log = np.concatenate(accuracy_log, axis=0)
            ax2.plot(acc_log[:,0],acc_log[:,1])
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Iteration')
            plt.show()