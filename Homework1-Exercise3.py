#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:30:41 2021

@author: mnagara

Homework 1 - Exercise 3

Linear Regression
"""
import numpy as np
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# (a) Generate 50 points of y over the interval x = np.linspace(-1,1,50)
x = np.linspace(-1,1,50) 
beta = np.array([-0.001, 0.01, 0.55, 1.5, 1.2])
ep = np.random.normal(0, 0.2**2, 50)
y = beta[0]*eval_legendre(0,x) + beta[1]*eval_legendre(1,x) + \
    beta[2]*eval_legendre(2,x) + beta[3]*eval_legendre(3,x) +\
        beta[4]*eval_legendre(4,x) + ep
        
plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
plt.title('Scatter plot of Y', fontsize=24)
plt.show()

# (b) Derive what X should be
X = np.column_stack((eval_legendre(0,x), eval_legendre(1,x), \
                     eval_legendre(2,x), eval_legendre(3,x), \
                     eval_legendre(4,x)))
    
# X = np.column_stack((np.ones(50), x, x**2, x**3, x**4))
#(c) Write a Python code to compute the solution. Overlay your predicted curve with the scattered plot.
theta = np.linalg.lstsq(X, y, rcond=None)[0]
t     = np.linspace(-1, 1, 200);
yhat  = theta[0]*eval_legendre(0,t) + theta[1]*eval_legendre(1,t) + \
        theta[2]*eval_legendre(2,t) + theta[3]*eval_legendre(3,t) + \
        theta[4]*eval_legendre(4,t)
        
plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.plot(t,yhat,'r', linewidth=3)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
plt.title('The solution fit over the data', fontsize=24)
plt.show()

#(d) Repeat with outliers
idx = [10,16,23,37,45]
y[idx] = 5
theta = np.linalg.lstsq(X, y, rcond=None)[0]
t     = np.linspace(-1, 1, 200);
yhat  = theta[0]*eval_legendre(0,t) + theta[1]*eval_legendre(1,t) + \
        theta[2]*eval_legendre(2,t) + theta[3]*eval_legendre(3,t) + \
        theta[4]*eval_legendre(4,t)
        
plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.plot(t,yhat,'r', linewidth=3)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
plt.title('The solution in the presence of outliers', fontsize=24)
plt.show()

# (f) Solve the program using linprog
# c = [0 0 0 0 0 1 1 ... 1]
c = np.hstack((np.zeros(5), np.ones(50)))
# b = [y -y]
b = np.hstack((y, -y))
# A = [phi I, -phi -I]
# Where phi_n = [1 L1(x_n) L2(x_n) L3(x_n) L4(x_n)]
phi = np.column_stack((eval_legendre(0,x), eval_legendre(1,x), eval_legendre(2,x), eval_legendre(3,x), eval_legendre(4,x)))
A = np.vstack((np.hstack((phi, -np.eye(50))), np.hstack((-phi, -np.eye(50)))))

res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None), method='revised simplex')
theta_prime = res.x
yhat  = theta_prime[0]*eval_legendre(0,t) + theta_prime[1]*eval_legendre(1,t) + \
        theta_prime[2]*eval_legendre(2,t) + theta_prime[3]*eval_legendre(3,t) + \
        theta_prime[4]*eval_legendre(4,t)
plt.figure(figsize=(12,8))
plt.scatter(x,y)
plt.plot(t,yhat,'m', linewidth=3)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
plt.title('The solution in the presence of outliers', fontsize=24)
plt.show()