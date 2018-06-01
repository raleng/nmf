#!/usr/bin/env python3
import logging
import os

# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np

from utils import convergence_check, distance, nndsvd, save_results


def normalize(norm, h):
    """ Normalizing with H """

    if norm == 'l1':
        norms = np.sum(h, 1)
    elif norm == 'l2':
        norms = np.sqrt(np.sum(h**2, 1))
    else:
        raise NameError('Don\'t recognize norm: {}'.format(norm))

    return norms


def w_update(distance_type, x, w, h, wh, lambda_w, norm):
    """ MUR Update and normalization """

    # Update step
    if distance_type:
        # w = w * ((x / (wh+1e-9)) @ h.T)
        # w /= np.ones((x.shape[0], x.shape[1])) @ h.T

        # Alternate update?
        b = np.ones((x.shape[0], x.shape[1])) @ h.T
        a = w * ((x / (wh+1e-9)) @ h.T)
        w = 2 * a / (b + np.sqrt(b * b + 4 * lambda_w * a))
    else:
        w = w * (x @ h.T) / (wh @ h.T + lambda_w * w + 1e-9)

    # Normalizing
    w = w * normalize(norm, h)

    return w


def h_update(distance_type, x, w, h, wh, lambda_h1, lambda_h2, norm):
    """ MUR Update with normalization """

    # Update step
    if distance_type:
        # h = h * (w.T @ (x / (wh+1e-9)))
        # h /= w.T @ np.ones((x.shape[0], x.shape[1]))

        # Alternative Update?
        c = h * (w.T @ (x / (wh+1e-9)))
        d = lambda_h1 * np.ones(h.shape) + w.T @ np.ones((x.shape[0], x.shape[1]))
        h = 2 * c / (d + np.sqrt(d * d + 4 * lambda_h2 * c))
    else:
        # here was simply lambda_h
        h = h * (w.T @ x) / (w.T @ wh + lambda_h1 * h + 1e-9)

    # Normalizing
    h = h / normalize(norm, h)[:, None]

    return h


def mur(x, k, *, distance_type='kl', norm='l2', max_iter=100000, tol1=1e-3, tol2=1e-3,
        lambda_w=0.0, lambda_h1=0.0, lambda_h2=0.0, save_dir='./results/',
        save_file='nmf'):
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
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_str = os.path.join(save_dir, save_file)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'distance_type': distance_type,
                       'max_iter': max_iter,
                       'tol1': tol1,
                       'tol2': tol2,
                       'lambda_w': lambda_w,
                       'lambda_h1': lambda_h1,
                       'lambda_h2': lambda_h2,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    # tol_precision = len(str(tol)) if tol < 1 else 0
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # make sure data is positive; should be anyways but data could contain small
    # negative numbers due to rounding errors and such
    if np.min(x) < 0:
        x += abs(np.min(x))
        logging.info('Data elevated.')

    # normalizing
    x = x/np.max(x[:])

    # initialize W and H
    w, h = nndsvd(x, k)

    # precomputing w @ h
    # saves one computation each iteration
    wh = w @ h

    obj_history = [distance(x, wh, type=distance_type)]

    print('Entering Main Loop.')
    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(distance_type, x, w, h, wh, lambda_w, norm)
        h = h_update(distance_type, x, w, h, w @ h, lambda_h1, lambda_h2, norm)
        wh = w @ h

        # Iteration info
        obj_history.append(distance(x, wh, type=distance_type))
        logging.info('[{}]: {:.{}f}'.format(i, obj_history[-1], tol_precision))

        # Check convergence; save and break iteration
        if convergence_check(obj_history[-1], obj_history[-2], tol1, tol2):
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
