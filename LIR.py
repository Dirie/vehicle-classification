# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from data_loader import *

def hypothesis(w,x):
    h = np.dot(w,x.T)
    return h
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
        
        for n in np.arange(N):
            xn = x[n,:][np.newaxis,:]
            yn = y[n,:][:,np.newaxis]
            h = hypothesis(w,xn)
            Dj = np.dot((h - yn),xn)
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

h = hypothesis(w,xt).T

e = np.sum((h-yt)**2)/yt.shape[0]
print('Error: ' + str(e))
