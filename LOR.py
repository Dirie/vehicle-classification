# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from data_loader import *

classes = ['phishing','non-phishing']
cls_index = [1,0]

    
def hypothesis(w,x):
    z = np.dot(w,x.T)
    h = 1.0/(1.0 + np.exp(-z))
    return h
def is_converged(w,wp,ep=1e-8):
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
        
        for n in np.arange(N):
            xn = x[n,:][np.newaxis,:]
            yn = y[n,:][:,np.newaxis]
            h = hypothesis(w,xn)
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
xo = xt.copy()
x = normalize_data(x)
xt = normalize_data(xt)

np.random.seed(8)
w = np.random.random(x.shape[1])[np.newaxis,:] * 0.1
w = learn_theta(x, w)

h = hypothesis(w,xo).T
h[h<=0.5]=0
h[h>0.5]=1
e = np.sum((h-yt)**2)/yt.shape[0]
print('Error: ' + str(e))
