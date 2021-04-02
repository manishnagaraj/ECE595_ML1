#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:24:46 2021

@author: mnagara

Homework 5
"""
# %%%%%%%%%
import numpy as np
import random 
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy.matlib
import math

from tqdm import tqdm
#%%%%%%%%%%
'''
Exercise 2
'''
# 1 is heads, 0 is tails
# Number of coins
n = 1000
# Probability of each
p = 0.5
# Number of coin throws for each  coin
t = 10
# Number of runs of experiment
r = 100000

'''
(a)What is the probability of getting a head for coin
'''
mu1=0.5
mumin=0.5
murand=0.5

'''
(b) repeat this entire experiment for 100, 000 runs to get 100,000 instances of V1, Vrand and
Vmin. Plot the histograms of the distributions of these three random variables.
'''

V1 = np.zeros(r)
Vmin = np.zeros(r)
Vrand = np.zeros(r)

for run in tqdm(range(r)):
    coins=np.zeros((n,t))
    for i in range(n):
        coins[i] =np.random.binomial(1, p, t)
    
    V_val = np.sum(coins==1, axis=1)/t
    V1[run] = V_val[0]
    Vrand[run] = random.choice(V_val)
    Vmin[run] = min(V_val)
    
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,8))
ax1.hist(V1)
ax1.set_xlabel(r'Value $V_1$')
ax1.set_ylabel('Number of trials')
ax1.set_title(r'$V_1$')
ax2.hist(Vrand)
ax2.set_xlabel(r'Value $V_{rand}$')
ax2.set_title(r'$V_{rand}$')
ax3.hist(Vmin)
ax3.set_xlabel(r'Value $V_{min}$')
ax3.set_title(r'$V_{min}$')
fig.show()

'''
(c) Using (b), plot the estimated P(|V1 ??µ?1| > \eps), P(|Vrand ??µ?rand| > eps) and P(|Vmin ??µ?min| > eps), together
with the Hoeffding??s bound 2 exp(-2(\eps**2)*N), for \eps = 0, 0.05, 0.1, ..., 0.5
'''
epsilon = np.arange(0, 0.501, 0.005) 
P_V1 = np.zeros(len(epsilon))
P_Vmin = np.zeros(len(epsilon))
P_Vrand = np.zeros(len(epsilon))
Hoff_bnd = np.zeros(len(epsilon))

for idx,eps in enumerate(epsilon):
    Hoff_bnd[idx] = 2*math.exp(-2*(eps**2)*t)
    P_V1[idx] = ((abs(V1-mu1) > eps).sum())/r
    P_Vmin[idx] = ((abs(Vmin-mumin) > eps).sum())/r
    P_Vrand[idx] = ((abs(Vrand-murand) > eps).sum())/r

plt.figure(figsize=(12,8))
plt.plot(epsilon, Hoff_bnd, 'k--', 
         epsilon, P_V1, 'r-*',
         epsilon, P_Vrand, 'b',
         epsilon, P_Vmin, 'g')
plt.xlabel(r'$\epsilon$')
plt.ylabel('Probability')
plt.legend(['Hoeffdings Bound', r'$V_1$', r'$V_{rand}$', r'$V_{min}$'])
plt.title(r'$P(\vert V_i - \mu_i \vert) > \epsilon$')
plt.grid()
plt.show()

#%%
''' 
Exercise 3 
'''
p_bern = 0.5
epsilon_bern = 0.01
Nset = np.round(np.logspace(2,5,100)).astype(int)
x = np.zeros((10000,Nset.size))
prob_simulate  = np.zeros(100)
prob_chernoff = np.zeros(100)
prob_hoeffding = np.zeros(100)

beta = 1 + ((0.5+epsilon_bern)*math.log((0.5+epsilon_bern),2)) + ((0.5-epsilon_bern)*math.log((0.5-epsilon_bern),2))
# Generate experiment for each N
for i in range(Nset.size):
    Num = Nset[i]
    x[:,i] = stats.binom.rvs(Num, p_bern, size=10000)/Num
    # calculate the probability of P(X_N-mu)
    prob_simulate[i]  = np.mean((np.abs(x[:,i]-p_bern)>=epsilon_bern).astype(float))
    # Chernoff
    prob_chernoff[i] = math.pow(2,(-1*beta*Num))
    # Hoeffding
    prob_hoeffding[i] = np.exp(-2*Num*(epsilon_bern**2))
    
plt.figure(figsize=(12,8))
plt.loglog(Nset, prob_simulate, 'x')
plt.loglog(Nset, prob_chernoff)
plt.loglog(Nset, prob_hoeffding, '*')
plt.xlabel('N')
plt.ylabel('Probability')
plt.legend([r'$P(\bar{X_{N}}-\mu \geq \epsilon)$', 'Chernoff Bound', 'Hoeffding Bound'])
plt.title('Exercise 3(b)')
plt.grid()
plt.show()
