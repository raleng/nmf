#!/usr/bin/python3

import argparse
import numpy as np
import scipy.io as sio
import mur_funcs

#Function definitions
def dist_euclid( X, WdotH ):
    """Euclidian distance"""
    value = 0.5 * np.sum( (X - WdotH)**2 )
    return value

def dist_kl( X, W, H ):
    """Kullback-Leibler divergence"""
    value = X * np.log( X / np.dot(W,H) )
    #value(~isfinite(value))=0
    value = np.sum( value - X + np.dot(W,H) )
    return value

def mur( X, k, kl=False ):
    """ NMF with MUR 
    
    Expects following arguments:
    X -- 2D Data
    k -- number of components
    """
    #Parameters
    maxiter = 100
    tol = 1e-5
    s = 1e-5
    savestr = 'nmf_mur_' + str(k) + '_' + str(kl)
    savestr = 'delme' 
    if np.min(X) < 0:
        X = X + abs(np.min(X))
        print('Daten werden hochgesetzt.')
    
    X = X/np.max(X[:])
    
    xdim = X.shape[0]
    samples = X.shape[1]
    
    #W = np.abs(np.random.randn(xdim,k))
    #H = np.abs(np.random.randn(k,samples))
    print('Loading initial matrices.')
    inimat = sio.loadmat('k4_ini.mat')
    W = inimat['W_ini']
    H = inimat['H_ini']
    print(W.flags)
    print(H.flags)

    WdotH = np.dot(W,H) 
    if kl:
        print('Using Kullback-Leibler divergence.')
        objhistory = [dist_kl(X,W,H)]
    else:
        print('Using euclidian distance.')
        #objhistory = [dist_euclid(X,WdotH)]
        objhistory = [mur_funcs.dist_euclid(X,WdotH)]
    
    for i in range(maxiter):
        begobj = objhistory[-1]
    
        if (i==maxiter-1):
            np.savez(savestr,W=W,H=H,i=i,objhistory=objhistory)
            print('Max iteration. Results saved in ' + savestr)
    
        if kl:
            H = H * ( np.dot(W.T, X/( np.dot(W,H)+1e-9 ) ) / np.sum(W,0) ) 
            W = W * ( np.dot(H, (X/(np.dot(W,H)+1e-9 )).T / np.sum(H,1) ) )
        else:
            #H = H * np.dot( W.T, X ) / ( np.dot( W.T, WdotH ) + 1e-9)
            #W = W * np.dot( X, H.T ) / ( np.dot( W, np.dot(H,H.T) ) + 1e-9)  
            W, H = mur_funcs.WH_update(X,W,H)
            print(W.flags)
            print(H.flags)
    
        norms = np.sqrt(np.sum(H.T**2,0))
        H = H/norms[:,None]
        W = W*norms
        WdotH = np.dot(W,H) 
        if kl:
            newobj = dist_kl(X,W,H)
        else:
            #newobj = dist_euclid(X,WdotH)
            newobj = mur_funcs.dist_euclid(X,WdotH)
    
        print('[' + str(i) + ']: ' + str(newobj))
        objhistory.append(newobj)

        #Konvergenzkriterium 1
        if newobj < tol:
            print('Algorithmus konvergiert (1)')
            np.savez(savestr,W=W,H=H,i=i,objhistory=objhistory)
            print('Results saved in ' + savestr) 
            break
    
        #Konvergenzkriterium 2
        if newobj >= begobj-s:
            print('Algorithmus konvergiert (2)')
            np.savez(savestr,W=W,H=H,i=i,objhistory=objhistory)
            print('Results saved in ' + savestr) 
            break

if __name__ == '__main__':
    # mur.py executed as script

    #Parsing arguments
    p = argparse.ArgumentParser()
    p.add_argument('-f', default='', type=str, dest='filestr')
    p.add_argument('-k', default=1, type=int)
    p.add_argument('-kl', action='store_true')
    args = p.parse_args()
    k = args.k

    # loading data from file
    X = np.fromfile(args.filestr, np.float32)
    X = np.reshape(X, (332*332*79, 9), order='F')

    # call function
    mur( X, k, args.kl )
