#!/usr/bin/python3
import begin
import logging
import numpy as np
import scipy.io as sio
import mur_funcs
from misc import loadme

def mur(X, k, *, kl=False, maxiter=100000, alpha_W=0, alpha_H=0,
        save_dir="./results/", save_file="nmf_default"):
    """ NMF with MUR

    Expects following arguments:
    X -- 2D Data
    k -- number of components
    """
    #Parameters
    tol = 1e-3
    s = 1e-3
    savestr = '{}{}_{}_{}'.format(save_dir, save_file, k, ('KL' if kl else 'EU'))
    #savestr = './results/nmf_mur_' + str(k) + '_' + str(kl)
    #savestr = './results/delme2'

    if np.min(X) < 0:
        X = X + abs(np.min(X))
        logging.info('Data elevated.')

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
        logging.info('Using Kullback-Leibler divergence.')
        objhistory = [mur_funcs.dist_kl(X, WdotH)]
    else:
        logging.info('Using euclidian distance.')
        #objhistory = [dist_euclid(X,WdotH)]
        objhistory = [mur_funcs.dist_euclid(X, WdotH)]

    for i in range(maxiter):
        begobj = objhistory[-1]

        if (i==maxiter-1):
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            logging.warning('Max iteration. Results saved in {}'.format(savestr))

        if kl:
            W, H = mur_funcs.WH_update_kl(X, W, H)
        else:
            W, H = mur_funcs.WH_update_euclid(X, W, H, alpha_W, alpha_H)

        norms = np.sqrt(np.sum(H.T**2, 0))
        H = H / norms[:, None]
        W = W * norms
        WdotH = np.dot(W, H)
        if kl:
            newobj = mur_funcs.dist_kl(X, WdotH)
        else:
            newobj = mur_funcs.dist_euclid(X, WdotH)

        logging.info('[{}]: {}'.format(i, newobj))
        objhistory.append(newobj)

        #Konvergenzkriterium 1
        if newobj < tol:
            logging.warning('Algorithm converged (1)')
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            logging.warning('Results saved in {}'.format(savestr))
            break

        #Konvergenzkriterium 2
        if newobj >= begobj-s:
            logging.warning('Algorithm converged (2)')
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            logging.warning('Results saved in {}'.format(savestr))
            break

        if i%100 == 0:
            np.savez(savestr, W=W, H=H, i=i, objhistory=objhistory)
            logging.warning('Saved on iteration {} in {}'.format(i, savestr))

@begin.start(auto_convert=True, lexical_order=True, short_args=False)
@begin.logging
def main(
        load_file='',
        load_var='LOAD_MSOT',
        features=1,
        kl=False,
        alpha_W=0,
        alpha_H=0,
        maxiter=100000,
        save_file='nmf_default',
        save_dir='./results/',
        ):

    if load_var == 'LOAD_MSOT':
        X = loadme.msot(load_file)
    else:
        X = loadme.pet(load_file, load_var)
        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
            logging.info('Data was 3D. Reshaped to 2D.')

    mur(X, k=features, kl=kl, maxiter=maxiter, alpha_W=alpha_W,
        alpha_H=alpha_H, save_dir=save_dir, save_file=save_file)
