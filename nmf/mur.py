#!/usr/bin/env python3
import logging
import os

from collections import namedtuple
import numpy as np
import numpy.linalg as la

from .utils import convergence_check, distance, nndsvd, save_results


def normalize(w):

    norm = la.norm(w, axis=0, ord=1)
    wn = w / norm

    return wn


def w_update(distance_type, x, w, h, wh, lambda_w=0.):
    """ MUR Update """

    # Update step
    if distance_type == 'kl':
        a = w * ((x / (wh+1e-9)) @ h.T)
        b = np.ones_like(x) @ h.T
        w = 2 * a / (b + np.sqrt(b**2 + 4 * lambda_w * a))
    elif distance_type == 'eu':
        w = w * (x @ h.T) / (wh @ h.T + lambda_w * w + 1e-9)
    else:
        raise KeyError('Unknown distance type.')

    return w


def h_update(distance_type, x, w, h, wh, lambda_h=0.):
    """ MUR Update """

    # Update step
    if distance_type == 'kl':
        c = h * (w.T @ (x / (wh+1e-9)))
        d = 0 * np.ones(h.shape) + w.T @ np.ones_like(x)
        h = 2 * c / (d + np.sqrt(d**2 + 4 * lambda_h * c))
    elif distance_type == 'eu':
        h = h * (w.T @ x) / (w.T @ wh + lambda_h * h + 1e-9)
    else:
        raise KeyError('Unknown distance type.')

    return h


def mur(x, k, *, distance_type='kl', min_iter=100, max_iter=100000, tol1=1e-5, tol2=1e-5,
        lambda_w=0.0, lambda_h=0.0, nndsvd_init=(False, 'zero'), save_dir='./results/'):
    """ Non-negative matrix factorization using multiplicative update rules

    Following the papers:
    - Lee, Seung: Learning the parts of objects by non-negative matrix factorization, 1999
    - Lee, Seung: Algorithms for non-negative matrix factorization, 2001 

    Expects following arguments:
    x -- 2D Data
    k -- number of components

    Accepts keyword arguments:
    distance_type -- STRING: 'eu' for Euclidean, 'kl' for Kullback Leibler
    min_iter -- INT: minimum number of iterations
    max_iter -- INT: maximum number of iterations
    tol1 -- FLOAT: convergence tolerance
    tol2 -- FLOAT: convergence tolerance
    lambda_w -- FLOAT: regularization parameter for w-Update
    lambda_h -- FLOAT: regularization parameter for h-Update
    nndsvd_init -- Tuple(BOOL, STRING): if BOOL = True, use NNDSVD-type STRING
    save_dir -- STRING: folder to which to save
    """

    # experiment parameters and results namedtuple
    Experiment = namedtuple('Experiment', 'method components distance_type nndsvd_init max_iter tol1 tol2 lambda_w lambda_h')
    Results = namedtuple('Results', 'w h i obj_history experiment')

    # experiment parameters
    experiment = Experiment(method='mur',
                            components=k,
                            distance_type=distance_type,
                            nndsvd_init=nndsvd_init,
                            max_iter=max_iter,
                            tol1=tol1,
                            tol2=tol2,
                            lambda_w=lambda_w,
                            lambda_h=lambda_h)


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

    # initialize W and H
    if nndsvd_init[0]:
        w, h = nndsvd(x, k, variant=nndsvd_init[1])
    else:
        w = np.abs(np.random.randn(x.shape[0], k))
        h = np.abs(np.random.randn(k, x.shape[1]))

    # precomputing w @ h
    wh = w @ h

    # initialize obj_history
    obj_history = [distance(x, wh, distance_type)]

    logging.info('Entering Main Loop.')
    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(distance_type, x, w, h, wh, lambda_w)
        h = h_update(distance_type, x, w, h, w @ h, lambda_h)
        wh = w @ h

        # Iteration info
        obj_history.append(distance(x, wh, distance_type))
        print('[{}]: {:.{}f}'.format(i, obj_history[-1], tol_precision))

        # Check convergence; save and break iteration
        if i > min_iter:
            converged = convergence_check(obj_history[-1], obj_history[-2], tol1, tol2)
            if converged:
                results = Results(w=w, h=h, i=i, obj_history=obj_history, experiment=experiment)
                logging.warning('Converged.')
                return results

        # save every XX iterations
        # if i % 50 == 0 and saving:
        #    save_results(save_str, w, h, i, obj_history, experiment_dict)

    else:
        logging.info('Max iteration reached.')

    results = Results(w=w, h=h, i=i, obj_history=obj_history, experiment=experiment)
    return results
