import numpy as np
import logging

def dist_euclid(X, WdotH):
    """Euclidian distance"""
    value = 0.5 * np.sum( (X - WdotH)**2 )
    return value

def dist_kl(X, WdotH):
    """Kullback-Leibler divergence"""
    value = X * np.log( X / WdotH )
    value = np.where(np.isinf(value), 0, value)
    value = np.sum( value - X + WdotH )
    return value

def WH_update_euclid(X, W, H, alpha_W, alpha_H):
    H = H * ( W.T @ X ) / ( W.T @ ( W @ H ) + alpha_H * H + 1e-9 )
    W = W * ( X @ H.T ) / ( W @ ( H @ H.T ) + alpha_W * W + 1e-9 )
    return W, H

def WH_update_kl(X, W, H):
    H = H * ((W.T @ (X / ((W @ H) + 1e-9))) / np.sum(W, 0)[:, np.newaxis])
    W = W * (H @ (X / ((W @ H) + 1e-9)).T / np.sum(H, 1)[:, np.newaxis]).T
    return W, H
