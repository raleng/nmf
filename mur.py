#!/usr/bin/python3

import argparse
import numpy as np
import scipy.io as sio
import mur_funcs
from misc import loadme

def mur(X, k, *, kl=False, maxiter=100000, alpha_W=0, alpha_H=0,
        dir_str="./results/", file_str="nmf_default"):
    """ NMF with MUR

    Expects following arguments:
    X -- 2D Data
    k -- number of components
    """
    #Parameters
    tol = 1e-3
    s = 1e-3
    savestr = dir_str + file_str + '_' + str(k) + '_' + ('KL' if kl else 'EU')
    #savestr = './results/nmf_mur_' + str(k) + '_' + str(kl)
    #savestr = './results/delme2'
    if np.min(X) < 0:
        X = X + abs(np.min(X))
        print('Daten werden hochgesetzt.')

    X = X/np.max(X[:])

    xdim = X.shape[0]
    samples = X.shape[1]

    W = np.abs(np.random.randn(xdim, k))
    H = np.abs(np.random.randn(k, samples))
    #print('Loading initial matrices.')
    #inimat = sio.loadmat('/home/ralf/uni/data/msot-matlab/k4_ini.mat')
    #W = inimat['W_ini']
    #H = inimat['H_ini']

    WdotH = np.dot(W,H)
    if kl:
        print('Using Kullback-Leibler divergence.')
        objhistory = [mur.funcs.dist_kl(X, WdotH)]
    else:
        print('Using euclidian distance.')
        #objhistory = [dist_euclid(X,WdotH)]
        objhistory = [mur_funcs.dist_euclid(X, WdotH)]

    for i in range(maxiter):
        begobj = objhistory[-1]

        if (i==maxiter-1):
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            print('Max iteration. Results saved in ' + savestr)

        if kl:
            W, H = mur_funcs.WH_update_kl(X, W, H)
        else:
            W, H = mur_funcs.WH_update_euclid(X, W, H, alpha_W, alpha_H)

        norms = np.sqrt(np.sum(H.T**2, 0))
        H = H/norms[:, None]
        W = W*norms
        WdotH = np.dot(W, H)
        if kl:
            newobj = mur_funcs.dist_kl(X, W, H)
        else:
            #newobj = dist_euclid(X,WdotH)
            newobj = mur_funcs.dist_euclid(X, WdotH)

        print('[' + str(i) + ']: ' + str(newobj))
        objhistory.append(newobj)

        #Konvergenzkriterium 1
        if newobj < tol:
            print('Algorithmus konvergiert (1)')
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            print('Results saved in ' + savestr)
            break

        #Konvergenzkriterium 2
        if newobj >= begobj-s:
            print('Algorithmus konvergiert (2)')
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            print('Results saved in ' + savestr)
            break

        if i%100 == 0:
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            print('Saved on iteration ' + str(i) + ' in ' + savestr)

if __name__ == '__main__':
    # mur.py executed as script

    #Parsing arguments
    p = argparse.ArgumentParser()
    
    #File loading MSOT (don't use -v)
    p.add_argument('-f', default='', type=str, dest='file_name')
    #File loading MAT (use -f AND -v)
    p.add_argument('-v', default='LOAD_MSOT', type=str, dest='var_name')

    p.add_argument('-k', default=1, type=int)
    p.add_argument('-kl', action='store_true')
    p.add_argument('-i', default=100000, type=int, dest='maxiter')
    p.add_argument('-aW', default=0, type=float, dest='alpha_W')
    p.add_argument('-aH', default=0, type=float, dest='alpha_H')
    p.add_argument('--file-str', default='nmf_default', type=str,
                   dest='file_str')
    p.add_argument('--dir-str', default='./results/', type=str,
                   dest='dir_str')
    args = p.parse_args()
    k = args.k

    # loading data from file
    if args.var_name == 'LOAD_MSOT':
        #X = np.fromfile(args.file_name, np.float32)
        #X = np.reshape(X, (332*332*79, 9), order='F')
        X = loadme.msot(args.file_name)
    else:
        X = loadme.pet(args.file_name, args.var_name)
        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
            print('Data was 3D. Reshaped to 2D.')

    # call function
    mur(X, k, kl=args.kl, maxiter=args.maxiter, alpha_W=args.alpha_W,
        alpha_H=args.alpha_H, file_str=args.file_str, dir_str=args.dir_str)
