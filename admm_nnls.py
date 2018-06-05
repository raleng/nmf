# system imports
import os
# noinspection PyUnresolvedReferences
import better_exceptions

# math imports
from math import sqrt
import numpy as np
from scipy import optimize

# personal imports
import fcnnls
from utils import convergence_check, distance, nndsvd, save_results

# TODO scaling! normalize W, H accordingly


def initialize(data, features):
    """ Initializing variables """
    # w = np.abs(np.random.randn(dims[0], features))
    # h = np.abs(np.random.randn(features, dims[1]))
    w, h = nndsvd(data, features)
    x = w @ h
    alpha_x = np.zeros_like(x)
    return x, w, h, alpha_x


def w_update(x, h, alpha_x, lambda_w, rho, *, use_fcnnls=False):
    """ ADMM update of W """

    x_mu = x + 1/rho * alpha_x
    x_mu = (x_mu > 0) * x_mu
    if np.any(x_mu < 0):
        print('[w]: Something neg.')
        print(np.min(x_mu))
    a = np.concatenate((sqrt(rho/2) * h.T, sqrt(lambda_w) * np.eye(h.shape[0])))
    b = np.concatenate((sqrt(rho/2) * x_mu.T, np.zeros((h.shape[0], x.shape[0]))))
    # A = np.concatenate(h.T, sqrt(2*lambda_w) * np.eye(h.shape[0]))
    # b = np.concatenate(x.T, np.zeros((h.shape[0], x.shape[0])))

    if use_fcnnls:
        w = fcnnls.fcnnls(a, b)
    else:
        w = np.zeros((a.shape[1], b.shape[1]))
        for i in range(b.shape[1]):
            w[:, i], _ = optimize.nnls(a, b[:, i])

    return w.T


def h_update(x, w, alpha_x, lambda_h, rho, *, use_fcnnls=False):
    """ ADMM update of H """

    x_mu = x + 1/rho * alpha_x
    x_mu = (x_mu > 0) * x_mu
    if np.any(x_mu < 0):
        print('[h]: Something neg.')
        print(np.min(x_mu))
    a = np.concatenate((sqrt(rho/2) * w, sqrt(lambda_h) * np.ones((1, w.shape[1]))))
    b = np.concatenate((sqrt(rho/2) * x_mu, np.zeros((1, x.shape[1]))))
    # A = np.concatenate(w, sqrt(2*lambda_h) * np.eye(w.shape[1]))
    # b = np.concatenate(x, np.zeros(w.shape[1], x.shape[1]))

    if use_fcnnls:
        h = fcnnls.fcnnls(a, b)
    else:
        h = np.zeros((a.shape[1], b.shape[1]))
        for i in range(b.shape[1]):
            h[:, i], _ = optimize.nnls(a, b[:, 1])

    return h


def x_update(v, wh, alpha_x, rho):
    """ ADMM update of X 
    
    Following Update formula from Sun & Févotte for beta = 1 (Kullback-Leibler)
    """

    value = rho * wh - alpha_x - 1
    x = value + np.sqrt(value**2 + 4 * rho * v)
    x /= (2 * rho)

    return x


def alpha_update(x, wh, alpha_x, rho):
    """ ADMM update dual variables """
    alpha_x += rho * (x - wh)
    return alpha_x


def admm_nnls(v, k, *, rho=1, use_fcnnls=False, lambda_w=0, lambda_h=0, min_iter=10,
         max_iter=100000, tol1=1e-5, tol2=1e-5, save_dir='./results/'):
    """ NMF with ADMM

    Expects following arguments:
    v -- 2D data
    k -- number of components

    Accepts keyword arguments:
    rho -- FLOAT: ADMM parameter
    lambda_w -- FLOAT: regul   arization parameter for W
    lambda_h -- FLOAT: regularization parameter for H
    max_iter -- INT: maximum number of iterations (default: 100000)
    save_dir -- STRING: folder to which to save
    """
    
    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_mur_{feat}_{lambda_w}_{lambda_h}'.format(
        feat=k,
        lambda_w=lambda_w,
        lambda_h=lambda_h,
    )
    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'max_iter': max_iter,
                       'rho': rho,
                       'lambda_w': lambda_w,
                       'lambda_h': lambda_h
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # initialize variables
    x, w, h, alpha_x = initialize(v, k)

    # initial distance value
    obj_history = [distance(v, x)]

    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(x, h, alpha_x, lambda_w, rho, use_fcnnls=use_fcnnls)
        h = h_update(x, w, alpha_x, lambda_h, rho, use_fcnnls=use_fcnnls)
        wh = w @ h

        x = x_update(v, wh, alpha_x, rho)
        alpha_x = alpha_update(x, wh, alpha_x, rho)

        # Iteration info
        obj_history.append(distance(v, x))
        print('[{}]: {:.{}f}'.format(i, obj_history[-1], tol_precision))

        # Check convergence; save and break iteration
        if i > min_iter:
            converged = convergence_check(obj_history[-1], obj_history[-2], tol1, tol2)
            if converged:
                save_results(save_str, w, h, i, obj_history, experiment_dict)
                print('Converged.')
                break

        # save every XX iterations
        if i % 100 == 0:
            save_results(save_str, w, h, i, obj_history, experiment_dict)

    else:
        # save on max_iter
        save_results(save_str, w, h, max_iter, obj_history, experiment_dict)
        print('Max iteration reached.')
