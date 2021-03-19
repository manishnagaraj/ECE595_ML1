#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:48:25 2021

@author: mnagara
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from numpy.matlib import repmat

#####################################################################################
######## EXERCISE 3 #################################################################

'''
Download the dataset from the course website. There are two classes with class labels yn = 1 and yn = 0.
(a) Show that the logistic regression loss is given by
(b) Introduce a regularization term 
Use CVXPY to minimize this loss function for the dataset I provided. 
'''

def convt(fld):
    return -float(fld[:-1]) if fld.endswith(b'-') else float(fld)

X0_og = np.matrix(np.loadtxt('data/homework4_class0.txt', converters={0: convt, 1: convt}))
X1_og = np.matrix(np.loadtxt('data/homework4_class1.txt', converters={0: convt, 1: convt}))
on = np.ones((X0_og.shape[0],1))
X0 = np.hstack((X0_og,on))
on = np.ones((X1_og.shape[0],1))
X1 = np.hstack((X1_og,on))
Y0 = np.zeros((X0.shape[0],1))
Y1 = np.ones((X1.shape[0],1))


x0_0 = np.zeros(50)
x0_1 = np.zeros(50)
x1_0 = np.zeros(50)
x1_1 = np.zeros(50)
for i in range(50):
    x0_0[i] = X0[i,0]
    x0_1[i] = X0[i,1]
    x1_0[i] = X1[i,0]
    x1_1[i] = X1[i,1]
    
x = np.vstack((X0, X1))
y = np.vstack((Y0,Y1))

lambd       = 0.0001

N = X0.shape[0] + X1.shape[0]

theta       = cvx.Variable((3,1))
loss        = - cvx.sum(cvx.multiply(y, x @ theta)) \
               + cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((N,1)), x @ theta]), axis=1 ) )
reg         = cvx.sum_squares(theta)
prob        = cvx.Problem(cvx.Minimize(loss/N + lambd*reg))
prob.solve()
w = theta.value

'''
(c) Scatter plot the data points by marking the two classes in two colors. Then plot the decision boundary.
'''
xbnd = np.linspace(-5,10,100)
ybnd = -(w[0]*xbnd + w[2])/w[1]
plt.figure(figsize=(8,8))
plt.scatter(x0_0, x0_1, c='r', marker='o')
plt.scatter(x1_0, x1_1, c='b', marker='x')
plt.plot(xbnd, ybnd, 'g', linewidth=2.0)
plt.legend(['Decision Boundary','Class0', 'Class1'])
plt.title('Logistic Regression')
plt.show()

'''
(d) Repeat (c) using Bayesian
'''
# calculate mean and covariance
mu_0 = np.mean(X0_og, axis=0)
sigma_0 = np.cov(X0_og.T)
mu_1 = np.mean(X1_og, axis=0)
sigma_1 = np.cov(X1_og.T)

sigma_0inv = np.linalg.inv(sigma_0)
sigma_1inv = np.linalg.inv(sigma_1)

def get_ll(data, mean, cov_inv, cov):
    data = data - mean
    ll = -0.5*np.matmul(np.matmul(data, cov_inv), data.T) -0.5*np.log(np.linalg.det(cov))
    return ll

xb = np.linspace(-5,10,100)
yb = np.linspace(-5,10,100)
pred = np.zeros((100, 100))
for i, x1 in enumerate(xb):
    for j, y1 in enumerate(yb):
        dat = np.column_stack([x1, y1])
        llr = get_ll(dat, mu_1, sigma_1inv, sigma_1) - get_ll(dat, mu_0, sigma_0inv, sigma_0)
        pred[i,j] = 1.0 if llr>0 else 0.0
        
plt.figure(figsize=(8,8))
plt.scatter(x0_0, x0_1, c='r', marker='o')
plt.scatter(x1_0, x1_1, c='b', marker='x')       
plt.contour(xb,yb,pred)
plt.legend(['Class0', 'Class1'])
plt.title('Bayesian Decision Boundary')
plt.show()

##################################################################################################
####################### Exercise 4 ###############################################################

'''
(a) Construct Kernel for the data matrix
K[m,n] = exp{-||x[m]-x[n]||2}
'''
K = np.zeros((N, N))
h=1
for i in range(N):
    for j in range(N):
       K[i,j] = np.exp(-np.sum(np.square(x[i,:]-x[j,:]))/h)
        
print(K[47:52, 47:52])

lambd_2 = 0.001

alpha = cvx.Variable((100,1))
loss_knl = -cvx.sum(cvx.multiply(y, K@alpha))\
           +cvx.sum(cvx.log_sum_exp( cvx.hstack([np.zeros((N,1)), K@alpha]), axis=1 ) )
reg_knl = cvx.quad_form(alpha, K)
prob_knl = cvx.Problem(cvx.Minimize(loss_knl/N + lambd_2*reg_knl))
prob_knl.solve()

w_alp = alpha.value

print(w_alp[:2])

# Evaluate on a grid of testing sites
xset = np.linspace(-5,10,100)
yset = np.linspace(-5,10,100)
output = np.zeros((100,100))
for i in range(100):
  for j in range(100):
    data = repmat( np.array([xset[i], yset[j], 1]).reshape((1,3)), N, 1)
    sub = data - x
    k_sub = np.exp(-np.sum(np.square(sub)/h, axis=1))
    output[i,j] = np.dot(w_alp.T, k_sub).item()

plt.figure(figsize=(8,8))
plt.scatter(x0_0, x0_1, c='r', marker='o')
plt.scatter(x1_0, x1_1, c='b', marker='x')  
plt.contour(xset, yset, output>0.5, linewidths=1, colors='k')
plt.legend(['Class0', 'Class1'])
plt.title('Kernel Method')
plt.show()