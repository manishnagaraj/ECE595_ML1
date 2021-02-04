#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:17:36 2021

@author: mnagara

Homework 1: Exercise 2

Gaussian Whitening

"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power
import math
# (a) 
step = 0.01
x1, x2 = np.mgrid[-1:5+step:step, 0:10+step:step]
num = (2*(x1**2)) + (2*(x2**2)) + (-2*x1*x2) + (4*x1) - (20*x2) + 56
den = math.sqrt(4 * math.pi * math.pi * 3)
y = np.exp(-num / 6) / den
plt.figure(figsize=(12,8))
plt.contour(x1,x2,y)
plt.xlabel('X1', fontsize=24)
plt.ylabel('X2',fontsize=24)
plt.title('PDF of X',fontsize=24)
# plt.grid()
plt.show()


# (c)
#(i) Use numpy.random.multivariate_normal to draw 5000 random samples from the 2D standard
# normal distribution, and make a scatter plot of the data point using matplotlib.pyplot.scatter.
X = np.random.multivariate_normal([0, 0], [[1,0],[0,1]], size=[5000])
plt.figure(figsize=(12,8))
plt.scatter(X[:,0], X[:,1])
plt.xlabel('X1', fontsize=24)
plt.ylabel('X2',fontsize=24)
plt.title('Randomly sampled values from 2D Standard Normal Distribution',fontsize=24)
plt.show()

# Apply the affine transformation you derived in part (b)(iv) to the data points, and make a
# scatter plot of the transformed data points. Now check your answer by using the Python function
# numpy.linalg.eig to obtain the trasformation and making a new scatter plot of the transformed
# data points
A = np.array([[math.sqrt(3), 1], [math.sqrt(3), -1]]) / math.sqrt(2)
b = np.array([[2, 6]] * 5000).T
Y = (np.matmul(A, X.T) + b).T
plt.figure(figsize=(12,8))
plt.scatter(Y[:,0], Y[:,1])
plt.xlabel('X1', fontsize=24)
plt.ylabel('X2',fontsize=24)
plt.title('Samples after affine transformation',fontsize=24)
plt.show()

Sigma = np.array([[2,1],[1,2]])
w, v = np.linalg.eig(Sigma)
w = w * np.eye(w.shape[0])
A_np = np.matmul(v, np.sqrt(w))
Y_np = (np.matmul(A_np, X.T) + b).T
plt.figure(figsize=(12,8))
plt.scatter(Y[:,0], Y[:,1])
plt.xlabel('X1', fontsize=24)
plt.ylabel('X2',fontsize=24)
plt.title('Samples using np.linalg.eig',fontsize=24)
plt.show()



#



# (c) (ii) Apply  the  affine  transformation  you  derived  in  part  (b)(iv)  
# to  the  data  points,  and  make  ascatter plot of the transformed data points. 


