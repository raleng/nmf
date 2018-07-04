#!/usr/bin/env python3
import logging
import os

# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np
import numpy.linalg as la

from utils import convergence_check, distance, nndsvd, save_results


def normalize(w):

    norm = la.norm(w, axis=0, ord=1)
    wn = w / norm

    #hn = np.zeros_like(h)
    #for i in range(h.shape[1]):
    #    hn[:, i] = h[:, i] * norm

    return wn #, hn
# def normalize(norm, h):
#     """ Normalizing with H """
#
#     if norm == 'l1':
#         norms = np.sum(h, 1)
#     elif norm == 'l2':
#         norms = np.sqrt(np.sum(h**2, 1))
#     else:
#         raise NameError('Don\'t recognize norm: {}'.format(norm))
#
#     return norms


def w_update(distance_type, x, w, h, wh, lambda_w=0):
    """ MUR Update and normalization """

    # Update step
    if distance_type == 'kl':
        pass
        w = w * ((x / (wh+1e-9)) @ h.T)
        w /= np.ones_like(x) @ h.T

        # Alternate update?
        # b = np.ones((x.shape[0], x.shape[1])) @ h.T
        # a = w * ((x / (wh+1e-9)) @ h.T)
        # w = 2 * a / (b + np.sqrt(b * b + 4 * lambda_w * a))
    elif distance_type == 'eu':
        w = w * (x @ h.T) / (wh @ h.T + lambda_w * w + 1e-9)
        # w = w * (x @ h.T) / (wh @ h.T + 1e-9)
    else:
        raise KeyError('Unknown distance type.')

    return w


def h_update(distance_type, x, w, h, wh, lambda_h=0):
    """ MUR Update with normalization """

    # Update step
    if distance_type == 'kl':
        pass
        h = h * (w.T @ (x / (wh+1e-9)))
        h /= w.T @ np.ones_like(x)

        # Alternative Update?
        # c = h * (w.T @ (x / (wh+1e-9)))
        # d = lambda_h1 * np.ones(h.shape) + w.T @ np.ones((x.shape[0], x.shape[1]))
        # h = 2 * c / (d + np.sqrt(d * d + 4 * lambda_h2 * c))
    elif distance_type == 'eu':
        h = h * (w.T @ x) / (w.T @ wh + lambda_h * h + 1e-9)

        # h = h * (w.T @ x) / (w.T @ wh + 1e-9)
    else:
        raise KeyError('Unknown distance type.')

    return h


def mur(x, k, *, distance_type='kl', min_iter=100, max_iter=100000, tol1=1e-5, tol2=1e-5,
        lambda_w=0.0, lambda_h=0.0, nndsvd_init=(False, 'zero'), save_dir='./results/'):
    """ NMF with MUR

    Expects following arguments:
    x -- 2D Data
    k -- number of components

    Accepts keyword arguments:
    kl -- BOOL: if True, use Kullback Leibler, else Euclidean
    norm -- STRING: what norm to use (l1 or l2)
    max_iter -- INT: maximum number of iterations
    tol1 -- FLOAT: convergence tolerance
    tol2 -- FLOAT: convergence tolerance
    lambda_w -- FLOAT: regularization parameter for w-Update
    lambda_h -- FLOAT: regularization parameter for h-Update
    nndsvd_init -- BOOL: if True, use NNDSVD initialization
    save_dir -- STRING: folder to which to save
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_mur_{feat}_{dist}_{l_w}_{l_h}'.format(
        feat=k,
        dist=distance_type,
        l_w=lambda_w,
        l_h=lambda_h,
    )
    if nndsvd_init[0]:
        save_name += '_nndsvd{}'.format(nndsvd_init[1][0])
    else:
        save_name += '_random'
    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'distance_type': distance_type,
                       'max_iter': max_iter,
                       'tol1': tol1,
                       'tol2': tol2,
                       'lambda_w': lambda_w,
                       'lambda_h': lambda_h,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    # tol_precision = len(str(tol)) if tol < 1 else 0
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # make sure data is positive; should be anyways but data could contain small
    # negative numbers due to rounding errors and such
    if np.min(x) < 0:
        amount = abs(np.min(x))
        x += amount
        logging.info('Data elevated by {}.'.format(amount))

    # normalizing
    # x = x/np.max(x[:])

    # initialize W and H
    if nndsvd_init[0]:
        w, h = nndsvd(x, k, variant=nndsvd_init[1])
    else:
        w = np.abs(np.random.randn(x.shape[0], k))
        h = np.abs(np.random.randn(k, x.shape[1]))

    # precomputing w @ h
    # saves one computation each iteration
    wh = w @ h

    obj_history = [distance(x, wh, distance_type)]

    print('Entering Main Loop.')
    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(distance_type, x, w, h, wh, lambda_w)
        # w = normalize(w)
        h = h_update(distance_type, x, w, h, w @ h, lambda_h)
        # Normalizing
        # h = h / normalize(norm, h)[:, None]
        wh = w @ h

        # Iteration info
        obj_history.append(distance(x, wh, distance_type))
        print('[{}]: {:.{}f}'.format(i, obj_history[-1], tol_precision))

        # Check convergence; save and break iteration
        if i > min_iter:
            converged = convergence_check(obj_history[-1], obj_history[-2], tol1, tol2)
            if converged:
                save_results(save_str, w, h, i, obj_history, experiment_dict)
                logging.warning('Converged.')
                break

        # save every XX iterations
        if i % 100 == 0:
            save_results(save_str, w, h, i, obj_history, experiment_dict)

    else:
        # save on max_iter
        save_results(save_str, w, h, max_iter, obj_history, experiment_dict)
        logging.info('Max iteration reached.')
