#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:45:15 2021

@author: mnagara

Homework 1: Exercise 1

Histogram and Cross-Validation

"""

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

# Let µ = 0 and sigma = 1 so that X ~ N (0, 1). Plot fX(x) using matplotlib.pyplot.plot for the range
# x in [?3, 3]. Use matplotlib.pyplot.savefig to save your figure.

plt.figure(figsize=(12,8))
x = np.linspace(-3, 3, 1000)
y = 1/(2*np.pi)*np.exp(-(x**2)/2)
plt.plot(x,y)
plt.xlabel('x', fontsize=24)
plt.ylabel('fx(x)',fontsize=24)
plt.title('X ~ N(0,1)',fontsize=24)
plt.grid()
plt.show()

#Let us investigate the use of histograms in data visualization

s = np.random.normal(size=[1000])
plt.figure(figsize=(12,8))
plt.hist(s, bins=4)
plt.title('Histogram with 4 bins',fontsize=24)
plt.show()

plt.figure(figsize=(12,8))
plt.hist(s, bins=1000)
plt.title('Histogram with 1000 bins',fontsize=24)
plt.show()

#Use scipy.stats.norm.fit to estimate the mean and standard deviation of your data
mu, sigma = scipy.stats.norm.fit(s)
print('The fit mean:{} and fit variance:{}'.format(mu, sigma))

#Plot the fitted gaussian curve on top of the two histogram plots using scipy.stats.norm.pdf.
plt.figure(figsize=(12,8))
plt.hist(s, bins=4, density=True)
fit_plt = scipy.stats.norm.pdf(x, mu, sigma)
plt.plot(x, fit_plt, linewidth=3)
plt.title('Histogram with 4 bins with fitted Gaussian curve',fontsize=24)
plt.show()

plt.figure(figsize=(12,8))
plt.hist(s, bins=1000, density=True)
plt.plot(x, fit_plt, linewidth=3)
plt.title('Histogram with 1000 bins with fitted Gaussian curve',fontsize=24)
plt.show()

# Plot Jb(h) with respect to m the number of bins, for m = 1, 2, ..., 200
n=1000 
m = np.arange(1, 200)
J = np.zeros((199))
for i in range(199):
    hist, bins = np.histogram(s, bins=m[i])
    h = n/m[i]
    J[i] = 2/((n-1)*h)-((n+1)/((n-1)*h))*np.sum((hist/n)**2)
plt.figure(figsize=(12,8))
plt.plot(m,J, linewidth=2)
plt.xlabel('Number of bins (m)', fontsize=24)
plt.ylabel('Cross validation score (J(h))',fontsize=24)
plt.title('Cross Validation Risk Estimation',fontsize=24)
plt.grid()
plt.show()  
  
# Find the m* that minimizes Jb(h), plot the histogram of your data with that m* 
best_no_bins = np.argmin(J)
print('Best number of bins:{}'.format(best_no_bins))
plt.figure(figsize=(12,8))
plt.hist(s, bins=best_no_bins)
plt.title('Histogram with m* bins',fontsize=24)
plt.show()

# Plot the Gaussian curve fitted to your data on top of your histogram.
plt.figure(figsize=(12,8))
plt.hist(s, bins=best_no_bins, density=True)
# fit_best = scipy.stats.norm.pdf(binsbest, mu, sigma)
plt.plot(x, fit_plt, linewidth=3)
plt.title('Histogram with m* bins with fitted Guassian curve',fontsize=24)
plt.show()
