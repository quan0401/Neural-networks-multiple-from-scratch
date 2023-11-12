import numpy as np

def relu(Z: np.ndarray): 
    Z[Z <= 0] = 0
    return Z

def sigmoid(Z: np.ndarray): 
    return 1/ (1 + np.exp(-Z))

def d_sigmoid(dA: np.ndarray, cache): 
    '''
    dA is computed result from previous derivative(s)
    '''
    Z = cache
    a = sigmoid(Z)
    dZ = dA * a * (1 - a)
    return  dZ

def d_relu(dA: np.ndarray, cache): 
    '''
    dA is computed result from previous derivative(s)
    '''
    Z = cache
    # Make sure to copy not referencing
    dZ = np.array(dA, copy=True)
    dZ[Z <=0] = 0
    return dZ
    
    