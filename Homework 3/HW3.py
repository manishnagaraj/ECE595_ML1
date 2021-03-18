#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:41:12 2021

@author: mnagara

Homework 3
"""

import numpy as np
import matplotlib.pyplot as plt


#####################################################################################
######## EXERCISE 2 #################################################################

train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter = ','))

Y = plt.imread('data/cat_grass.jpg')/255

# cat -- 1 & grass -- 0
# (b) Estimate means, variances and priors

mu_1 = np.mean(train_cat, axis=1)
mu_0 = np.mean(train_grass, axis=1)
sigma_1 = np.cov(train_cat)
sigma_0 = np.cov(train_grass)
pi_1 = train_cat.shape[1]/(train_cat.shape[1] + train_grass.shape[1])
pi_0 = train_grass.shape[1]/(train_cat.shape[1] + train_grass.shape[1])

'''    
(c) Write a double for loop to loop through the pixels of the testing image. At each pixel location, consider
a 8 ×? 8 neighborhood. This will be the testing vector x ?? R
64. Dump this testing vector x into the
decision rule you proved in (a), and determine whether the testing vector belongs to Class 1 or Class
0. Repeat this for other pixel locations.
'''
M = Y.shape[0]
N = Y.shape[1]
prediction = np.zeros((M-8, N-8))


# LHS = -1/2 (x-mu_1).T sigma_1^(-1) (x-mu_1) + log(pi_1)


for i in range(M-8):
    for j in range(N-8):
        block = Y[i:i+8, j:j+8]
        x = block.reshape((64,1))
        # LHS = -1/2 (x-mu_1).T sigma_1^(-1) (x-mu_1) + log(pi_1) - 1/2 log(\sigma_1)
        sub_1 = x - mu_1
        sig_1_inv = np.linalg.inv(sigma_1)
        LHS = -0.5*np.matmul(sub_1.T, np.matmul(sig_1_inv, sub_1)).item() + np.log(pi_1) -0.5*np.log(np.linalg.det(sigma_1))
        # RHS = -1/2 (x-mu_0).T sigma_0^(-1) (x-mu_0) + log(pi_0) - 1/2 log(\sigma_0)
        sub_0 = x - mu_0
        sig_0_inv = np.linalg.inv(sigma_0)
        RHS = -0.5*np.matmul(sub_0.T, np.matmul(sig_0_inv, sub_0)).item() + np.log(pi_0) -0.5*np.log(np.linalg.det(sigma_0))
        
        if LHS > RHS :
            prediction[i][j]=1

plt.imshow(prediction, cmap='gray')

'''
(d) Consider the ground truth image truth.png. Report the mean absolute error (MAE) between your
prediction and the ground truth:
'''
Y_tr = plt.imread('data/truth.png')
truth = Y_tr[0:M-8, 0:N-8]
MAE = np.sum(np.abs(truth-prediction))/prediction.size

'''
(e) Go to the internet and download an image with similar content: an animal on grass or something like
that. Apply your classifier to the image, and submit your resulting mask. You probably do not have
the ground truth mask, so please just show the predicted mask. Does it perform well? If not, what
could go wrong? Write one to two bullet points to explain your findings. Please be brief.
'''
Y_test = plt.imread('data/animal_grass.jpeg')/255
Y_test_grayscale = np.mean(Y_test, axis=2)
M_test = Y_test_grayscale.shape[0]
N_test = Y_test_grayscale.shape[1]

mask = np.zeros((M_test-8, N_test-8))
for i in range(M_test-8):
    for j in range(N_test-8):
        block_2 = Y_test_grayscale[i:i+8, j:j+8]
        x = block_2.reshape((64,1))
        # LHS = -1/2 (x-mu_1).T sigma_1^(-1) (x-mu_1) + log(pi_1) - 1/2 log(\sigma_1)
        sub_1 = x - mu_1
        sig_1_inv = np.linalg.inv(sigma_1)
        LHS = -0.5*np.matmul(sub_1.T, np.matmul(sig_1_inv, sub_1)).item() + np.log(pi_1) -0.5*np.log(np.linalg.det(sigma_1))
        # RHS = -1/2 (x-mu_0).T sigma_0^(-1) (x-mu_0) + log(pi_0) - 1/2 log(\sigma_0)
        sub_0 = x - mu_0
        sig_0_inv = np.linalg.inv(sigma_0)
        RHS = -0.5*np.matmul(sub_0.T, np.matmul(sig_0_inv, sub_0)).item() + np.log(pi_0) -0.5*np.log(np.linalg.det(sigma_0))
        
        if LHS > RHS :
            mask[i][j]=1        
            
plt.imshow(mask, cmap='gray')

#######################################################################################
########################## EXERCISE 3 #################################################
'''
(b) Implement this likelihood ratio test rule for different values of tau . For every tau , compute the number
of true positives and the number of false positives. Then, we can define the probability of detection
pD(tau) and the probability of miss pF (tau) as:
'''
P_d_t = []
P_f_t = []

tau_MAP = (pi_0/pi_1)

# P_c1-----log(p(x|C1)) = -1/2 log(det(sigma_1)) -1/2 (x-mu_1).T 
# P_c0-----log(p(x|C0)) = -1/2 log(det(sigma_0)) -1/2 (x-mu_0).T 

# iterate over tau from log(0.01/0.99) ~ -5 to +5
tau_val = np.append(np.append(np.append(-1000, np.append(-100, range(-10, 10))), 100), 1000)
for tau in tau_val:
    # Predict  for each tau------
    curr_pred = np.zeros((M-8,N-8))
    for i in range(M-8):
        for j in range(N-8):
            block = Y[i:i+8, j:j+8]
            x = block.reshape((64,1))
            sub_1 = x - mu_1
            sig_1_inv = np.linalg.inv(sigma_1)
            P_c1 = -0.5*np.matmul(sub_1.T, np.matmul(sig_1_inv, sub_1)).item() -0.5*np.log(np.linalg.det(sigma_1))
            sub_0 = x - mu_0
            sig_0_inv = np.linalg.inv(sigma_0)
            P_c0 = -0.5*np.matmul(sub_0.T, np.matmul(sig_0_inv, sub_0)).item() -0.5*np.log(np.linalg.det(sigma_0))
            
            if (P_c1 - P_c0) > tau :
                curr_pred[i][j]=1
    
    # calulate total positives in ground truth -------- all pixels which are cat ---- ie all pixels with val = 1
    Total_positives = np.count_nonzero(truth>0.5)
    # calculate total negatives in ground truth ------ all pixels that are 0 
    Total_negatives = np.count_nonzero(truth<0.5)
    # true positives ---- predicted as 1, and actually 1
    true_positives = 0
    false_positives = 0
    # false_positives ---- predicted as 1, actually 0
    for i in range(M-8):
        for j in range(N-8):
            if curr_pred[i][j] == 1 and truth[i][j]>0.5 :
                true_positives = true_positives + 1
            elif curr_pred[i][j] == 1 and truth[i][j]<0.5 :
                false_positives = false_positives + 1
    P_d = (true_positives)/Total_positives
    P_f = (false_positives)/ Total_negatives
    
    P_d_t +=[P_d]
    P_f_t +=[P_f]

# plot roc --  P_d_t vs P_f_t
plt.figure(figsize=(12,8))
plt.plot(P_f_t, P_d_t, linewidth=3)
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.ylabel(r"$p_D(\tau)$", fontsize=24)
plt.xlabel(r"$p_F(\tau)$", fontsize=24)
plt.title("ROC",fontsize=24)
plt.grid()
plt.show()

'''
On your ROC curve, mark a red dot to indicate the operating point of the Bayesian decision rule.
'''
# Predict  for each tauMAP
MAP_pred = np.zeros((M-8,N-8))
for i in range(M-8):
    for j in range(N-8):
        block = Y[i:i+8, j:j+8]
        x = block.reshape((64,1))
        sub_1 = x - mu_1
        sig_1_inv = np.linalg.inv(sigma_1)
        P_c1 = -0.5*np.matmul(sub_1.T, np.matmul(sig_1_inv, sub_1)).item() -0.5*np.log(np.linalg.det(sigma_1))
        sub_0 = x - mu_0
        sig_0_inv = np.linalg.inv(sigma_0)
        P_c0 = -0.5*np.matmul(sub_0.T, np.matmul(sig_0_inv, sub_0)).item() -0.5*np.log(np.linalg.det(sigma_0))
        
        if (P_c1 - P_c0) > tau_MAP :
            MAP_pred[i][j]=1

# calulate total positives in ground truth -------- all pixels which are cat ---- ie all pixels with val = 1
Total_positives = np.count_nonzero(truth>0.5)
# calculate total negatives in ground truth ------ all pixels that are 0 
Total_negatives = np.count_nonzero(truth<0.5)
# true positives ---- predicted as 1, and actually 1
true_positives = 0
false_positives = 0
# false_positives ---- predicted as 1, actually 0
for i in range(M-8):
    for j in range(N-8):
        if MAP_pred[i][j] == 1 and truth[i][j]>0.5 :
            true_positives = true_positives + 1
        elif MAP_pred[i][j] == 1 and truth[i][j]<0.5 :
            false_positives = false_positives + 1
P_d_MAP = (true_positives)/Total_positives
P_f_MAP = (false_positives)/ Total_negatives
# plot roc --  P_d_t vs P_f_t
plt.figure(figsize=(12,8))
plt.plot(P_f_t, P_d_t, linewidth=3)
plt.plot(P_f_MAP, P_d_MAP, 'ro', linewidth=6)
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.ylabel(r"$p_D(\tau)$", fontsize=24)
plt.xlabel(r"$p_F(\tau)$", fontsize=24)
plt.legend(['ROC Curve', 'Bayesian Classifier'])
plt.title("ROC",fontsize=24)
plt.grid()
plt.show()

'''
(d) Implement a linear regression classifier for this problem, and plot the ROC curve
'''
# construct A 
A = np.vstack((train_cat.T, train_grass.T))
# construct b 
b = -1*np.ones((((train_cat.shape[1])+(train_grass.shape[1])),1))
# b of first half is +1 
b[0:train_cat.shape[1]] = 1
# \theta = inv(A.T,A)@A.T@b
theta = np.linalg.inv(A.T@A) @ A.T @ b

P_d_t_reg = []
P_f_t_reg = []
# Test this out for the image Y
for tau in range(-5000, 5000, 100):
    reg_pred = np.zeros((M-8, N-8))
    for i in range(M-8):
        for j in range(N-8):
            block = Y[i:i+8, j:j+8]
            x = block.reshape((64,1))
            pred = theta.T @ x
            if pred > tau :
                reg_pred[i][j] = 1
    true_positives = 0
    false_positives = 0
    for i in range(M-8):
        for j in range(N-8):
            if reg_pred[i][j] == 1 and truth[i][j]>0.5 :
                true_positives = true_positives + 1
            elif reg_pred[i][j] == 1 and truth[i][j]<0.5 :
                false_positives = false_positives + 1
    P_d = (true_positives)/Total_positives
    P_f = (false_positives)/ Total_negatives
    
    P_d_t_reg +=[P_d]
    P_f_t_reg +=[P_f]    

plt.figure(figsize=(12,8))
plt.plot(P_f_t, P_d_t, linewidth=3)
plt.plot(P_f_t_reg, P_d_t_reg, 'g',linewidth=3)
plt.xticks(np.arange(0.0, 1.1, 0.1))
plt.yticks(np.arange(0.0, 1.1, 0.1))
plt.ylabel(r"$p_D(\tau)$", fontsize=24)
plt.xlabel(r"$p_F(\tau)$", fontsize=24)
plt.legend(['ROC of Likelihood Decision Rule', 'ROC of Regression'])
plt.title("ROC",fontsize=24)
plt.grid()
plt.show()