#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:22:46 2021

@author: mnagara
"""
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

from models import LeNetBinary

class ReweightingAlgorithm():
    def __init__(self, hyperparameters, net, device, data_loader, test_loader):
        self.hyperparameters = hyperparameters
        self.net = net
        self.device = device
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.opt = torch.optim.SGD(self.net.params(),lr=self.hyperparameters["lr"])  
    
    def learn(self, plot_step):
        accuracy_log = []
        
        val_data = self.data_loader.dataset.data_val.to(self.device)
        val_labels = self.data_loader.dataset.labels_val.to(self.device)
        
        for i in tqdm(range(self.hyperparameters['num_iterations'])):
            self.net.train()
            # Line 2 get batch of data
            image, labels = next(iter(self.data_loader))
            image, labels = image.to(self.device), labels.to(self.device)
            # since validation data is small I just fixed them instead of building an iterator
            # initialize a dummy network for the meta learning of the weights
            meta_net = LeNetBinary(n_out=1)
            meta_net.load_state_dict(self.net.state_dict())
            meta_net = meta_net.to(self.device)
            
            # Lines 4 - 5 initial forward pass to compute the initial weighted loss
            y_f_hat  = meta_net(image)
            cost = F.binary_cross_entropy_with_logits(y_f_hat,labels, reduce=False)
            eps = Variable(torch.zeros(cost.size()).to(self.device), requires_grad=True)
            l_f_meta = torch.sum(cost * eps)
            
            meta_net.zero_grad()        
            
            # Line 6 perform a parameter update
            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
            meta_net.update_params(self.hyperparameters['lr'], source_params=grads)
            
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
            y_f_hat = self.net(image)
            cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
            l_f = torch.sum(cost * w)
    
            self.opt.zero_grad()
            l_f.backward()
            self.opt.step()
    
            
            if i % plot_step == 0:
                self.net.eval()
    
                acc = []
                for itr,(test_img, test_label) in enumerate(self.test_loader):
                    test_img = test_img.to(self.device)
                    test_label = test_label.to(self.device)
    
                    output = self.net(test_img)
                    predicted = (F.sigmoid(output) > 0.5).int()
    
                    acc.append((predicted.int() == test_label.int()).float())
    
                accuracy = torch.cat(acc,dim=0).mean()
                accuracy_log.append(np.array([i,accuracy])[None])       
    