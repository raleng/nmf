import numpy as np

def dist_euclid(X,WdotH):
    value = 0.5 * np.sum( (X - WdotH)**2 )
    return value

def WH_update(X,W,H):
    H = H * np.dot( W.T, X ) / ( np.dot( W.T, np.dot(W,H) ) + 1e-9)
    W = W * np.dot( X, H.T ) / ( np.dot( W, np.dot(H,H.T) ) + 1e-9)  
    return W, H
