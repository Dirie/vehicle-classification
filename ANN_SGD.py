# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:14:13 2017

@author: dirie
"""

import numpy as np
from data_loader import *
import matplotlib.pyplot as pl
def act(s):
    """
    activation function
    """
    u = 1.0 / (1.0 + np.exp(-s))
    return u
    
def is_converged(theta, theta_p, epsilon=1e-8):
    """
    checks convergence
    theta: theta at iteration t
    thetap: theta at iteration t-1
    epsilon: a small number used for thresholdin
    """
    L = len(theta)
    d = np.max(np.abs(theta[0]-theta_p[0]))
    for l in np.arange(L):
        d_temp = np.max(np.abs(theta[l]-theta_p[l]))
        if d_temp > d:
            d = d_temp
    if  d <= epsilon:
        return True
    else:
        return False
    
def ffnn_learn(x, y, alpha, theta):
    """
    learns and returns the parameters of ffnn using SGD
    """
    N = x.shape[0]
    L = len(theta)
    converged = False
    epoch = 0
    while not converged:
        mse = 0.0
        for n in np.arange(N): # for each observation
            xn = x[n, :][np.newaxis, :]
            yn = y[n, :][np.newaxis, :]
            s = dict()
            u = dict()
            s[0] = xn.copy()
            u[0] = s[0].copy()
            for l in np.arange(1, L+1): # apply forward propagation
                s[l] = np.dot(u[l-1], theta[l-1])
                u[l] = act(s[l])
                if l != L:
                    u[l][0,0] = 1.0                
            en = u[l] - yn # error of the network
            mse += np.sum(en**2)
            theta_p = theta.copy()
            for l in np.arange(L-1, -1, -1): # back propagation
                delta = (u[l+1] * (1.0 - u[l+1]) * en)
                theta[l] = theta[l] - alpha * np.dot(u[l].T, delta)
                if l != L-1:
                    theta[l][0,0] = 0.0                    
                # accumulate the error at the previous layer
                en = np.dot(en, theta[l].T)
            converged = is_converged(theta, theta_p)
            if converged:
                break;
        epoch += 1
        mse = mse / n
        msg = 'epoch: {:7d}, MSE: {:.6f}'.format(epoch, mse)
        print(msg)
    return theta

def ffnn_classify(x, theta):
    """
    classifies x according ffnn architecture
    """
    L = len(theta)
    s = dict()
    u = dict()
    s[0] = x.copy()
    u[0] = s[0].copy()
    for l in np.arange(1, L+1): # apply forward propagation
        s[l] = np.dot(u[l-1], theta[l-1])
        u[l] = act(s[l])
        if l != L:
            u[l][0,0] = 1.0
    return u[l]

def print_confusion_matrix(confusion_matrix):
    """
    This function performs formatted printing of the confusion matrix.
    """    
    rows, cols = confusion_matrix.shape
    print("Confusion Matrix:")
    sys.stdout.write('%-8s '%' ')
    for r in np.arange(0,rows):
        sys.stdout.write('%-8s '%get_key(r, targets))        
    sys.stdout.write('\n')
    for r in np.arange(0,rows):
        sys.stdout.write('%-8s '%get_key(r, targets))
        for c in np.arange(0,cols):
            sys.stdout.write('%-8d '%confusion_matrix[r,c])
        sys.stdout.write('\n')
    total_samples = np.sum(confusion_matrix)
    empirical_error = np.float((total_samples - np.trace(confusion_matrix))) / total_samples
    sys.stdout.write('empirical error = %.4f'%(empirical_error))
    sys.stdout.write('\n')
    
def get_key(key_value, targets):
    for key in targets.keys():
        if targets[key] == key_value:
            return key
    

def test_hypothesis(xt,yt, theta): 
    """
    This function tests the learned hypothesis, and produces
    the confusion matrix as the output. The diagonal elements of the confusion
    matrix refer to the number of correct classifications and non-diagonal elements
    fefer to the number of incorrect classifications.
    """
    h = ffnn_classify(xt, theta)
    h[h>0.5] = 1
    h[h<=0.5] = 0    
    num_classes = 2
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)   
    rows = h.shape[0]
    for i in np.arange(0,rows):
        confusion_matrix[int(yt[i]), int(h[i])] = confusion_matrix[int(yt[i]), int(h[i])] + 1
    return confusion_matrix
    
x,y,xt,yt = load_data()
xo = xt.copy()
x = normalize_data(x)
xt = normalize_data(xt)

np.random.seed(7)
theta = dict()
theta[0] = np.random.randn(x.shape[1], x.shape[1]) * 0.01
#theta[1] = np.random.randn(x.shape[1], x.shape[1]) * 0.01
#theta[2] = np.random.randn(x.shape[1], x.shape[1]) * 0.01
#theta[3] = np.random.randn(x.shape[1], x.shape[1]) * 0.01
#theta[4] = np.random.randn(x.shape[1], x.shape[1]) * 0.01
#theta[5] = np.random.randn(x.shape[1], x.shape[1]) * 0.01
theta[1] = np.random.randn(x.shape[1], y.shape[1]) * 0.01

alpha=0.1

theta = ffnn_learn(x, y, alpha, theta)


h = ffnn_classify(xt, theta)
e = np.sum((h - yt)**2)/xt.shape[0]


n = np.random.randint(xt.shape[0])
x = xt[n, :][np.newaxis,:]
y = yt[n, :][np.newaxis,:]

h = ffnn_classify(x, theta)
y_index = np.argmax(y)
h_index = np.argmax(h)
labels = ['phishing','non-phishing']
p = h.copy()
p[p<=0.5]=0
p[p>0.5]=1

y_index = int(y)
h_index = int(p)



title = 'Label : ' + labels[y_index] + ', Predicted : ' + labels[h_index] + '--'
print(title)
print('Error: {:.6f}'.format(e))



targets = {'phishing':1,
          'non-phishing':0
          }
          

##test the learned hypothesis
confusion_matrix = test_hypothesis(xt,yt, theta)
##print the confusion matrix
print_confusion_matrix(confusion_matrix)


    
