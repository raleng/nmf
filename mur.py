#!/usr/bin/python3
import begin
import logging
import numpy as np
import scipy.io as sio
from misc import loadme


def dist_euclid(X, WdotH):
    """Euclidean distance"""
    value = 0.5 * np.sum((X - WdotH)**2)
    return value


def dist_kl(X, WdotH):
    """Kullback-Leibler divergence"""
    value = X * np.log(X / WdotH)
    value = np.where(np.isinf(value), 0, value)
    value = np.sum(value - X + WdotH)
    return value


def WH_update_euclid(X, W, H, WdotH, alpha_W, alpha_H):
    """MUR Update with euclidean distance"""
    H = H * (W.T @ X) / (W.T @ WdotH + alpha_H * H + 1e-9)
    W = W * (X @ H.T) / (W @ (H @ H.T) + alpha_W * W + 1e-9)
    return W, H


def WH_update_kl(X, W, H, WdotH):
    """MUR Update with Kullback-Leibler divergence"""
    H = H * ((W.T @ (X / (WdotH + 1e-9))) / np.sum(W, 0)[:, np.newaxis])
    W = W * (H @ (X / (WdotH + 1e-9)).T / np.sum(H, 1)[:, np.newaxis]).T
    return W, H


def mur(X, k, *, kl=False, max_iter=100000, tol1=1e-3, tol2=1e-3, alpha_W=0, alpha_H=0,
        save_dir="./results/", save_file="nmf_default"):
    """ NMF with MUR

    Expects following arguments:
    X -- 2D Data
    k -- number of components

    Accepts keyword arguments:
    kl -- BOOL: if True, use Kullback Leibler, else Euclidean
    maxiter -- INT: maximum number of iterations
    tol1 -- FLOAT: convergence tolerance
    tol2 -- FLOAT: convergence tolerance
    alpha_W -- FLOAT: regularization parameter for W-Update
    alpha_H -- FLOAT: regularization parameter for H-Update
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = len(str(tol)) if tol < 1 else 0

    save_str = '{dir}{file}_{k}_{dist}'.format(dir=save_dir,
                                               file=save_file,
                                               k=k,
                                               dist=('KL' if kl else 'EU'),
                                               )

    # make sure data is positive; should be anyways but data could contain small
    # negative numbers due to rounding errors and such
    if np.min(X) < 0:
        X += abs(np.min(X))
        logging.info('Data elevated.')

    # normalizing
    X /= np.max(X[:])

    xdim = X.shape[0]
    samples = X.shape[1]

    # initializing W and H with random matrices
    W = np.abs(np.random.randn(xdim, k))
    H = np.abs(np.random.randn(k, samples))
    # print('Loading initial matrices.')
    # inimat = sio.loadmat('/home/ralf/uni/data/msot-matlab/k4_ini.mat')
    # W = inimat['W_ini']
    # H = inimat['H_ini']

    WdotH = W @ H
    if kl:
        logging.info('Using Kullback-Leibler divergence.')
        obj_history = [dist_kl(X, WdotH)]
    else:
        logging.info('Using euclidean distance.')
        obj_history = [dist_euclid(X, WdotH)]

    for i in range(max_iter):
        old_obj = obj_history[-1]

        if i == max_iter-1:
            np.savez(save_str, W=W, H=H, i=i, objhistory=obj_history)
            logging.warning('Max iteration. Results saved in {}'.format(save_str))

        # Update step
        if kl:
            W, H = WH_update_kl(X, W, H, WdotH)
        else:
            W, H = WH_update_euclid(X, W, H, WdotH, alpha_W, alpha_H)

        norms = np.sqrt(np.sum(H.T**2, 0))
        H = H / norms[:, None]
        W = W * norms
        WdotH = W @ H

        if kl:
            new_obj = dist_kl(X, WdotH)
        else:
            new_obj = dist_euclid(X, WdotH)

        logging.info('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # Check convergence
        break_true = True
        if new_obj < tol1:
            logging.warning('Algorithm converged (1)')
        elif new_obj >= old_obj-tol2:
            logging.warning('Algorithm converged (2)')
        else:
            break_true = False

        if break_true:
            np.savez(save_str, W=W, H=H, i=i, obj_history=obj_history)
            logging.warning('Results saved in {}'.format(save_str))
            break

        # save every XX iterations
        if i % 100 == 0:
            np.savez(save_str, W=W, H=H, i=i, obj_history=obj_history)
            logging.warning('Saved on iteration {} in {}'.format(i, save_str))


@begin.start(auto_convert=True, lexical_order=True, short_args=False)
@begin.logging
def main(load_file='',
         load_var='LOAD_MSOT',
         features=1,
         kl=False,
         max_iter=100000,
         tol1=1e-3,
         tol2=1e-3,
         alpha_W=0.0,
         alpha_H=0.0,
         save_file='nmf_default',
         save_dir='./results/',
         ):

    """ NMF with MUR """
    if load_var == 'LOAD_MSOT':
        X = loadme.msot(load_file)
    else:
        X = loadme.pet(load_file, load_var)
        if len(X.shape) == 3:
            X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
            logging.info('Data was 3D. Reshaped to 2D.')

    mur(X, features, kl=kl, max_iter=max_iter, tol1=tol1, tol2=tol2, alpha_W=alpha_W,
        alpha_H=alpha_H, save_dir=save_dir, save_file=save_file)
