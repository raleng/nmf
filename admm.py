# system imports
import os
# noinspection PyUnresolvedReferences
import better_exceptions

# math imports
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# personal imports
from utils import convergence_check, distance, nndsvd, save_results


def initialize(data, features, nndsvd_init):
    """ Initializing variables """

    if nndsvd_init[0]:
        w, h = nndsvd(data, features, variant=nndsvd_init[1])
    else:
        w = np.abs(np.random.randn(data.shape[0], features))
        h = np.abs(np.random.randn(features, data.shape[1]))

    x = w @ h

    w_p = w.copy()
    h_p = h.copy()

    alpha_x = np.zeros_like(x)
    alpha_w = np.zeros_like(w)
    alpha_h = np.zeros_like(h)

    return x, w, h, w_p, h_p, alpha_x, alpha_w, alpha_h


def w_update(x, h, w_p, alpha_x, alpha_w, rho):
    """ ADMM update of W """
    a = h @ h.T + np.eye(h.shape[0])
    b = h @ x.T + w_p.T + 1/rho * (h @ alpha_x.T - alpha_w.T)
    w = np.linalg.solve(a, b).T
    return w


def h_update(x, w, h_p, alpha_x, alpha_h, rho):
    """ ADMM update of H """
    a = w.T @ w + np.eye(w.shape[1])
    b = w.T @ x + h_p + 1/rho * (w.T @ alpha_x - alpha_h)
    h = np.linalg.solve(a, b)
    return h


def x_update(v, wh, alpha_x, rho, distance_type='kl'):
    """ ADMM update of X """
    if distance_type == 'kl':
        value = rho * wh - alpha_x - 1
        x = value + np.sqrt(value**2 + 4 * rho * v)
        x /= 2 * rho
    elif distance_type == 'eu':
        x = v
    else:
        raise KeyError('Unknown distance type.')

    return x


def prox(prox_type, mat_aux, dual, rho=None, lambda_=None):
    if prox_type == 'nn':
        # return np.maximum(mat_aux + 1/rho * dual, 0)
        diff = mat_aux - dual
        mat = np.where(diff < 0, 0, diff)
        return mat

    elif prox_type == 'l2n':
        n = mat_aux.shape[0]
        k = -np.array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)])
        offset = [-1, 0, 1]
        tikh = sp.diags(k, offset)  # .toarray()

        a = (lambda_ * tikh.T @ tikh - rho * sp.eye(n))
        b = rho * mat_aux - dual
        # a = 1/rho * (lambda_ * tikh.T @ tikh + rho * sp.eye(n))
        # b = mat_aux - dual
        mat = spla.spsolve(a, b)

        mat = np.where(mat < 0, 0, mat)
        return mat


def wh_p_update(w, h, alpha_w, alpha_h, rho):
    """ ADMM update of W_plus and H_plus """
    w_p = np.maximum(w + 1/rho * alpha_w, 0)
    h_p = np.maximum(h + 1/rho * alpha_h, 0)
    return w_p, h_p


def alpha_update(x, w, h, wh, w_p, h_p, alpha_x, alpha_w, alpha_h, rho):
    """ ADMM update dual variables """
    alpha_x = alpha_x + rho * (x - wh)
    alpha_h = alpha_h + rho * (h - h_p)
    alpha_w = alpha_w + rho * (w - w_p)
    return alpha_x, alpha_h, alpha_w


def admm(v, k, *, rho=1, distance_type='kl', reg_w=(0, 'nn'), reg_h=(0, 'l2n'),
         min_iter=10, max_iter=100000, tol1=1e-3, tol2=1e-3, nndsvd_init=(True, 'zero'),
         save_dir='./results/'):
    """ NMF with ADMM

    Expects following arguments:
    v -- 2D data
    k -- number of components

    Accepts keyword arguments:
    rho -- FLOAT:
    min_iter -- INT: minimum number of iterations (default: 10)
    max_iter -- INT: maximum number of iterations (default: 100000)
    tol1 -- FLOAT:
    tol2 -- FLOAT:
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_admm_{feat}_{rho}_{dist}_{lambda_w}:{prox_w}_{lambda_h}:{prox_h}'.format(
        feat=k,
        dist=distance_type,
        rho=rho,
        lambda_w=reg_w[0],
        prox_w=reg_w[1],
        lambda_h=reg_h[0],
        prox_h=reg_h[1],
    )
    if nndsvd_init[0]:
        save_name += '_nndsvd{}'.format(nndsvd_init[1][0])
    else:
        save_name += '_random'

    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'max_iter': max_iter,
                       'rho': rho,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    x, w, h, w_p, h_p, alpha_x, alpha_w, alpha_h = initialize(v, k, nndsvd_init)

    # initial distance value
    obj_history = [distance(v, w@h, distance_type=distance_type)]

    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(x, h, w_p, alpha_x, alpha_w, rho)
        h = h_update(x, w, h_p, alpha_x, alpha_h, rho)
        wh = w @ h

        x = x_update(v, wh, alpha_x, rho, distance_type)
        w_p = prox(reg_w[1], w, alpha_w, rho=rho, lambda_=reg_w[0])
        h_p = prox(reg_h[1], h.T, alpha_h.T, rho=rho, lambda_=reg_h[0])
        h_p = h_p.T
        # w_p, h_p = wh_p_update(w, h, alpha_w, alpha_h, rho)
        alpha_x, alpha_h, alpha_w, = alpha_update(x, w, h, wh, w_p, h_p, alpha_x, alpha_w,
                                                  alpha_h, rho)

        # Iteration info
        obj_history.append(distance(v, w_p@h_p, distance_type=distance_type))
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
