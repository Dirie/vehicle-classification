import numpy as np
import pandas as pd
import sys

classes = ['phishing','normal']
cls_index = [1,0]
          
def load_data():
    data = pd.read_csv('../data/dataset.csv', delimiter=',',dtype=float).values

    N,D = data.shape    
    X = np.zeros((N,D),dtype=float)
    Y = np.zeros((N,1),dtype=float)

    for c in np.arange(0,D-1):
        X[:,c+1] = data[:,c]
    X[:,0] = 1
    Y[:,0] = data[:,D-1]
    Y[Y==-1] = 0

    X_Positive = X[Y[:,0]==1,:]
    X_Negative = X[Y[:,0]==0,:]
    
    Y_Positive = Y[Y[:,0]==1,:]
    Y_Negative = Y[Y[:,0]==0,:] 
    
    Number_of_Positive, D_Postive = X_Positive.shape
    Number_of_Negative,D_Negative = X_Negative.shape
    
    index_positives = np.arange(0,Number_of_Positive)
    index_negatives = np.arange(0,Number_of_Negative)
    
    np.random.shuffle(index_positives)
    np.random.shuffle(index_negatives)
    
    num_training_positives = np.round(0.5 * Number_of_Positive)
    num_training_negatives = np.round(0.5 * Number_of_Negative)
    
        
    X_training = X_Positive[index_positives[0:num_training_positives],:]
    Y_trainig = Y_Positive[index_positives[0:num_training_positives],:]
    X_training = np.vstack((X_training, X_Negative[index_negatives[0:num_training_negatives],:]))
    Y_trainig = np.vstack((Y_trainig, Y_Negative[index_negatives[0:num_training_negatives],:]))


    X_testing = X_Positive[index_positives[num_training_positives:],:]
    Y_testing = Y_Positive[index_positives[num_training_positives:],:]
    X_testing = np.vstack((X_testing, X_Negative[index_negatives[num_training_negatives:],:]))
    Y_testing = np.vstack((Y_testing, Y_Negative[index_negatives[num_training_negatives:],:]))
    
    
    return X_training, Y_trainig, X_testing, Y_testing
    
def normalize_data(x):
    """
    this function normalizes the dataset by using Z-score.
    """
    m = np.mean(x, axis=0)
    s = np.std(x, axis=0)
    m[0] = 0.0
    s[0] = 1.0
    x = (x-m)
    x= x/s
    return x
