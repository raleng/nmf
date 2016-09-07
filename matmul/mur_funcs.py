import numpy as np

def dist_euclid(X, WdotH):
    """Euclidian distance"""
    value = 0.5 * np.sum( (X - WdotH)**2 )
    return value

def dist_kl(X, WdotH):
    """Kullback-Leibler divergence"""
    value = X * np.log( X / WdotH )
    #value(~isfinite(value))=0
    value = np.sum( value - X + WdotH )
    return value

def WH_update_euclid(X, W, H, alpha_W, alpha_H):
    print('Update without @')
    H = H * np.dot( W.T, X ) / ( np.dot( W.T, np.dot(W,H) ) + alpha_H*H + 1e-9)
    W = W * np.dot( X, H.T ) / ( np.dot( W, np.dot(H,H.T) ) + alpha_W*W + 1e-9)
    return W, H

def WH_update_euclid_at(X, W, H, alpha_W, alpha_H):
    print('Update with @')
    H = H * ( W.T @ X ) / ( W.T @ ( W @ H ) + alpha_H*H + 1e-9 )
    W = W * ( X @ H.T ) / ( W @ ( H @ H.T ) + alpha_W*W + 1e-9 )
    return W, H

def WH_update_kl(X, W, H):
    H = H * ( np.dot(W.T, X/( np.dot(W,H)+1e-9 ) ) / np.sum(W,0) )
    W = W * ( np.dot(H, (X/(np.dot(W,H)+1e-9 )).T / np.sum(H,1) ) )
    return W, H

alpha_H = 0.1
alpha_W = 0.2

X = np.random.rand(100,10)
W = np.random.rand(100,5)
H = np.random.rand(5,10)

W_neu, H_neu = WH_update_euclid(X, W, H, alpha_W, alpha_H)
W_neu_at, H_neu_at = WH_update_euclid_at(X, W, H, alpha_W, alpha_H)

if np.all(W_neu == W_neu_at):
    print('W passt.')
else:
    print('W passt nicht.')

if np.all(H_neu == H_neu_at):
    print('H passt.')
else:
    print('H passt nicht.')
