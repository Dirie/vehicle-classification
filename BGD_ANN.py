# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from data_loader import *




def sigmoid(w,x):
    z = np.dot(w,x.T)
    a = 1.0/(1.0 + np.exp(-z))
    return a
    
def sigmoid_prime(h,y):
    dj = (h * (1-h)*(h-y))
    return dj
    
def is_converged(w,wp,ep=1e-15):
    d = np.max(np.abs(w - wp))
    if d<=ep:
        return True
    else:
        return False

def learn_theta(x, w):
    N, D = x.shape
    alph = 0.1
    converged = False
    epoch =0
    while not converged:
        for l in np.arange(L):
            
            for n in np.arange(N):
                xn = x[n,:][np.newaxis,:]
                yn = y[n,:][:,np.newaxis]
                h = sigmoid(w,xn)
                J = (h * (1-h) * (h - yn))
                Dj = np.dot(J,xn)
                wp = w.copy()
                w = (w - (alph * Dj))
        converged = is_converged(w,wp)
        epoch +=1
        e = np.sum((h-yn)**2)
        print('epoch: ' + str(epoch) + ', Error:' + str(e))
    return w


    
x,y,xt,yt = load_data()
x = normalize_data(x)
w = np.random.random(x.shape[1])[np.newaxis,:]
w = learn_theta(x, w)

np.random.randint(xt.shape[0])

x = xt[n, :][np.newaxis,:]
y = yt[n, :][np.newaxis,:]

h = ffnn_classify(x, w)
y_index = np.argmax(y)
h_index = np.argmax(h)
cls_names = ['phishing','normal']



title = 'Label : ' + cls_names[y_index] + ', Predicted : ' + cls_names[h_index] + '--'
for c in np.arange(0, len(cls_names)):
    title += cls_names[c] + ' = ' + '{:.2f}'.format(h[0,c]) + ' , '
    
pl.title(title)
