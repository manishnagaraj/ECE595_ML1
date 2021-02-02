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

# (a) 
x1 = np.linspace(-1, 5, 1000)
x2 = np.linspace(0, 10, 1000)
y = np.zeros([1000, 1000])

for i in range(1000):
    for j in range(1000):
        # eqn = 2x1**2 + 2x2**2 - 2*x1*x2 +4*x1 -20*x2 +56
        eqn = (2*(x1[i]**2)) + (2*(x2[j]**2)) - (2*x1[i]*x2[j]) + (4*x1[i]) - (20*x2[j]) + 56
        y[i][j] = (1/np.sqrt(12*(np.pi**2)))*np.exp((-1/6)*eqn)
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
s = np.random.multivariate_normal([0, 0], [[1,0],[0,1]], size=[5000])
plt.figure(figsize=(12,8))
plt.scatter(s[:,0], s[:,1])
plt.xlabel('X1', fontsize=24)
plt.ylabel('X2',fontsize=24)
plt.title('Randomly sampled values from 2D Standard Normal Distribution',fontsize=24)
plt.show()

# Apply the affine transformation you derived in part (b)(iv) to the data points, and make a
# scatter plot of the transformed data points. Now check your answer by using the Python function
# numpy.linalg.eig to obtain the trasformation and making a new scatter plot of the transformed
# data points
mu = np.array([2,6])
Sigma = np.array([[2,1],[1,2]])
Sigma2 = fractional_matrix_power(Sigma, 0.5)
sy = np.dot(Sigma2, s.T) + numpy.matlib.repmat(mu, 5000, 1).T
plt.scatter(sy.T[:,0], sy.T[:,1])
plt.xlabel('X1', fontsize=24)
plt.ylabel('X2',fontsize=24)
plt.title('Samples after affine transformation',fontsize=24)
plt.show()

# 