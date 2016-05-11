#!/usr/bin/python3

import argparse
import numpy as np
import scipy.io as sio
import mur_funcs

def mur( X, k, kl=False, maxiter=100000, alpha_W=0, alpha_H=0 ):
    """ NMF with MUR 
    
    Expects following arguments:
    X -- 2D Data
    k -- number of components
    """
    #Parameters
    tol = 1e-3
    s = 1e-3
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
        objhistory = [mur.funcs.dist_kl(X,WdotH)]
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
            W, H = mur.funcs.WH_update_kl( X, W, H )
        else:
            W, H = mur_funcs.WH_update_euclid( X, W, H, alpha_W, alpha_H )
    
        norms = np.sqrt(np.sum(H.T**2,0))
        H = H/norms[:,None]
        W = W*norms
        WdotH = np.dot(W,H) 
        if kl:
            newobj = mur_funcs.dist_kl(X,W,H)
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
    p.add_argument('-i', default=100000, type=int, dest='maxiter')
    args = p.parse_args()
    k = args.k

    # loading data from file
    X = np.fromfile(args.filestr, np.float32)
    X = np.reshape(X, (332*332*79, 9), order='F')

    # call function
    mur( X, k, kl=args.kl, maxiter=args.maxiter )
