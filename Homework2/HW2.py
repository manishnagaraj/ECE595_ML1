#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:12:55 2021

@author: mnagara
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import csv

# Reading the male data
male_idx = []
male_bmi = []
male_stature = []
with open("data/male_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # Add your code here to process the data into usable form
    next(reader, None)
    for row in reader:
        male_idx+=[int(row[0])]
        male_bmi+=[float(row[1])/10]
        male_stature+=[float(row[2])/1000]
csv_file.close()

# Reading the female data
female_idx = []
female_bmi = []
female_stature = []
with open("data/female_train_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # Add your code here to process the data into usable form
    next(reader, None)
    for row in reader:
        female_idx+=[int(row[0])]
        female_bmi+=[float(row[1])/10]
        female_stature+=[float(row[2])/1000]
csv_file.close()

###########################################################################
# EXERCISE 1
#Print the first 10 elements of each column of the dataset

print('First 10 female BMI ',female_bmi[:10])    
print('First 10 female stature ',female_stature[:10])   
print('First 10 male BMI ',male_bmi[:10])
print('First 10 male stature ',male_stature[:10])      

###############################################################################
# EXERCISE 2
#(b) For the NHANES dataset, assign yn = +1 if the n-th sample is a male, and yn = -1 if the 
# n-th sample is a female. Implement your answer in (a) with Python to solve the problem.
#########
# Constructing Xn = [bmi stature]
No_values = len(male_bmi)+len(female_bmi)
dim = 3

X = np.zeros([len(male_bmi)+len(female_bmi), dim])
y = np.zeros(len(male_bmi)+len(female_bmi))
for idx in range(len(male_bmi)):
    y[idx]= 1
    X[idx][0] = 1
    X[idx][1] = male_bmi[idx]
    X[idx][2] = male_stature[idx]
idx_mal = idx+1
for idx in range(len(female_bmi)):
    y[idx+idx_mal] = -1
    X[idx+idx_mal][0] = 1
    X[idx+idx_mal][1] = female_bmi[idx]
    X[idx+idx_mal][2] = female_stature[idx] 

XtXi = np.linalg.inv(np.matmul(X.T, X))
theta = np.matmul(np.matmul(XtXi, X.T), y)
print(theta)
# (c) Repeat (b), but this time use CVXPY. Report your answer, and submit your code.
theta_cvx = cp.Variable(dim)
objective = cp.Minimize(cp.sum_squares(X*theta_cvx - y))
prob = cp.Problem(objective)
prob.solve()
beta = theta_cvx.value
print(beta)

# (e) Implement the gradient descent algorithm in Python

theta_gd = np.zeros(dim)
XtX = np.matmul(X.T, X)

training_loss = np.zeros(50000)
for k in range(50000):
    # d = nalbla e
    dJ = np.matmul(X.T,(np.matmul(X, theta_gd) - y))
    dd = dJ
    # alpha = dJ.dd/||Xd||^2
    alpha = np.matmul(dJ, dd)/np.matmul(np.matmul(XtX, dd), dd)
    theta_gd = theta_gd - alpha*dd
    training_loss[k] = np.linalg.norm(y-np.matmul(X, theta_gd))**2/len(y)
print(theta_gd)    

plt.figure(figsize=(12,8))
plt.semilogx(training_loss, linewidth=8)
plt.xlabel('Iteration', fontsize=24)
plt.ylabel('Training Loss',fontsize=24)
plt.title('Training loss using SGD',fontsize=24)
plt.grid()
plt.show()


# (f) Implement the gradient descent algorithm with momentum in Python

theta_gd_mom = np.zeros(dim)
training_loss = np.zeros(50000)
XtX = np.matmul(X.T, X)
dJ_old = np.zeros(dim)
beta = 0.9

for k in range(50000):
    dJ = np.matmul(X.T,(np.matmul(X, theta_gd_mom) - y))
    dd = beta*dJ_old + (1-beta)*dJ
    alpha = np.matmul(dJ, dd)/np.matmul(np.matmul(XtX, dd), dd)
    theta_gd_mom = theta_gd_mom - alpha*dd
    dJ_old = dJ
    training_loss[k] = np.linalg.norm(y-np.matmul(X, theta_gd_mom))**2/len(y)
print(theta_gd_mom)    

plt.figure(figsize=(12,8))
plt.semilogx(training_loss, linewidth=8)
plt.xlabel('Iteration', fontsize=24)
plt.ylabel('Training Loss',fontsize=24)
plt.title('Training loss using SGD with momentum',fontsize=24)
plt.grid()
plt.show()

###############################################################################
# Exercise 3
# (a) (i) Plot data points
#(ii) overlay with decision boundary

plt.figure(figsize=(12,8))
plt.scatter(X[:len(male_bmi),1], X[:len(male_bmi),2], c='b', marker='o')
plt.scatter(X[len(male_bmi):,1], X[len(male_bmi):,2], c='r', marker='.')

gx1=np.arange(0,8,0.001)
gx2= -(theta[0]+theta[1]*gx1)/(theta[2])

plt.plot(gx1,gx2, 'g', linewidth=3)
plt.legend(['Decision Boundary', 'Male datapoints', 'Female datapoints'])
plt.show()

# Reading testing data
# Reading the male data
male_idx_test = []
male_bmi_test = []
male_stature_test = []
with open("data/male_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # Add your code here to process the data into usable form
    next(reader, None)
    for row in reader:
        male_idx_test+=[int(row[0])]
        male_bmi_test+=[float(row[1])/10]
        male_stature_test+=[float(row[2])/1000]
csv_file.close()

# Reading the female data
female_idx_test = []
female_bmi_test = []
female_stature_test = []
with open("data/female_test_data.csv", "r") as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    # Add your code here to process the data into usable form
    next(reader, None)
    for row in reader:
        female_idx_test+=[int(row[0])]
        female_bmi_test+=[float(row[1])/10]
        female_stature_test+=[float(row[2])/1000]
csv_file.close()

Male_test = np.zeros([len(male_bmi_test), dim])
Male_test_label = np.zeros(len(male_bmi_test))
for idx in range(len(male_bmi_test)):
    Male_test_label[idx]= 1
    Male_test[idx][0] = 1
    Male_test[idx][1] = male_bmi_test[idx]
    Male_test[idx][2] = male_stature_test[idx]
idx_mal = idx+1

Female_test = np.zeros([len(female_bmi_test), dim])
Female_test_label = np.zeros(len(female_bmi_test))
for idx in range(len(male_bmi_test)):
    Female_test_label[idx]= -1
    Female_test[idx][0] = 1
    Female_test[idx][1] = female_bmi_test[idx]
    Female_test[idx][2] = female_stature_test[idx]
idx_mal = idx+1

#(b)
# False alarm of male (should be female but classified as a male)
pred_female = np.sign(np.matmul(theta, Female_test.T))
wrong_female = pred_female - Female_test_label
false_alarm_percentage = np.count_nonzero(wrong_female)/len(pred_female)*100
print('False Alarm Percentage {}%'.format(false_alarm_percentage))

# miss of male (should be male but classified as a female)
pred_male = np.sign(np.matmul(theta, Male_test.T))
wrong_male = pred_male - Male_test_label
miss_percentage = np.count_nonzero(wrong_male)/len(pred_male)*100
print('Miss Percentage {}%'.format(miss_percentage))

# actually male, declared male
True_positive = len(pred_male)-np.count_nonzero(wrong_male)
# actually female declare male
False_positive = np.count_nonzero(wrong_female)
# actually male declare female
False_negative = np.count_nonzero(wrong_male)

precision = (True_positive)/(True_positive+False_positive)
print('Precision is', precision)

recall = (True_positive)/(True_positive+False_negative)
print('Recall is', recall)

##########################################################################
########## Exercise 4 ######################################

lambd = np.arange(0.1, 10, 0.1)
normJ =  np.linalg.norm(np.matmul(X, theta)-y)**2 

t_lam_norm = np.zeros(len(lambd))
J_lam_norm = np.zeros(len(lambd))
# theta_lambd = nomrJ + lamda(norm(theta))
for idx, lam in enumerate(lambd):
    theta_lambd = cp.Variable(dim)
    objective_lambd = cp.Minimize((cp.sum_squares(X*theta_lambd - y)) + lam*cp.sum_squares(theta_lambd))
    prob_lam = cp.Problem(objective_lambd)
    prob_lam.solve()
    theta_soln = theta_lambd.value
    
    t_lam_norm[idx] = (np.linalg.norm(theta_soln))**2
    J_lam_norm[idx] = (np.linalg.norm(np.matmul(X, theta_soln)-y))**2
    
plt.figure(figsize=(12,8))
plt.plot(t_lam_norm, J_lam_norm , linewidth=3)
plt.ylabel(r"$\Vert X \theta_\lambda - y \Vert^2$", fontsize=24)
plt.xlabel(r"$\Vert \theta_\lambda \Vert^2$", fontsize=24)
plt.title(r"$\Vert X \theta_\lambda - y \Vert^2$ v/s $\Vert \theta_\lambda \Vert^2$",fontsize=24)
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(lambd, J_lam_norm, linewidth=3)
plt.ylabel(r"$\Vert X \theta_\lambda - y \Vert^2$", fontsize=24)
plt.xlabel(r"$ \lambda $", fontsize=24)
plt.title(r"$\Vert X \theta_\lambda - y \Vert^2$ v/s $ \lambda $",fontsize=24)
plt.grid()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(lambd, t_lam_norm, linewidth=3)
plt.ylabel(r"$\Vert \theta_\lambda \Vert^2$", fontsize=24)
plt.xlabel(r"$ \lambda $", fontsize=24)
plt.title(r"$\Vert \theta_\lambda \Vert^2$ v/s $ \lambda $",fontsize=24)
plt.grid()
plt.show()

