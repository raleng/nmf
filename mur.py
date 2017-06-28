#!/usr/bin/env python3
import logging
import os
from importlib import import_module

import begin
# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np
from misc import loadme


def dist_euclid(x, wh):
    """ Euclidean distance """
    value = 0.5 * np.sum((x - wh) ** 2)
    return value


def dist_kl(x, wh):
    """ Kullback-Leibler divergence """
    value = x * np.log(x / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - x + wh)
    return value


def normalize(norm, h):
    """ Normalizing with H """

    if norm == 'l1':
        norms = np.sum(h, 1)
    elif norm == 'l2':
        norms = np.sqrt(np.sum(h**2, 1))
    else:
        raise NameError('Don\'t recognize norm: {}'.format(norm))

    return norms


def w_update(kl, x, w, h, wh, alpha_w, norm):
    """ MUR Update and normalization """

    # Update step
    if kl:
        # w = w * ((x / (wh+1e-9)) @ h.T)
        # w /= np.ones((x.shape[0], x.shape[1])) @ h.T

        # Alternate update?
        b = np.ones((x.shape[0], x.shape[1])) @ h.T
        a = w * ((x / (wh+1e-9)) @ h.T)
        w = 2 * a / (b + np.sqrt(b * b + 4 * alpha_w * a))
    else:
        w = w * (x @ h.T) / (wh @ h.T + alpha_w * w + 1e-9)

    # Normalizing
    w = w * normalize(norm, h)

    return w


def h_update(kl, x, w, h, wh, alpha_h1, alpha_h2, norm):
    """ MUR Update with normalization """

    # Update step
    if kl:
        # h = h * (w.T @ (x / (wh+1e-9)))
        # h /= w.T @ np.ones((x.shape[0], x.shape[1]))

        # Alternative Update?
        c = h * (w.T @ (x /(wh+1e-9)))
        d = alpha_h1 * np.ones(h.shape) + w.T @ np.ones((x.shape[0], x.shape[1]))
        h = 2 * c / (d + np.sqrt(d * d + 4 * alpha_h2 * c))
    else:
        h = h * (w.T @ x) / (w.T @ wh + alpha_h * h + 1e-9)

    # Normalizing
    h = h / normalize(norm, h)[:, None]

    return h


def convergence_check(new, old, tol1, tol2):
    """ Checks the convergence criteria """
    convergence_break = True

    if new < tol1:
        logging.warning('Algorithm converged (1).')
    elif new >= old - tol2:
        logging.warning('Algorithm converged (2).')
    else:
        convergence_break = False

    return convergence_break


def mur(x, k, *, kl=False, norm='l2', max_iter=100000, tol1=1e-3, tol2=1e-3,
        alpha_w=0.0, alpha_h1=0.0, alpha_h2=0.0, save_dir='./results/', save_file='nmf'):
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
    alpha_W -- FLOAT: regularization parameter for w-Update
    alpha_H -- FLOAT: regularization parameter for h-Update
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_str = os.path.join(save_dir, save_file)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'kl': kl,
                       'max_iter': max_iter,
                       'tol1': tol1,
                       'tol2': tol2,
                       'alpha_w': alpha_w,
                       'alpha_h1': alpha_h1,
                       'alpha_h2': alpha_h2,
                       }


    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = len(str(tol)) if tol < 1 else 0


    # make sure data is positive; should be anyways but data could contain small
    # negative numbers due to rounding errors and such
    if np.min(x) < 0:
        x += abs(np.min(x))
        logging.info('Data elevated.')

    # normalizing
    # x /= np.max(x[:])
    x = x/np.max(x[:])

    # initializing w and h with random matrices
    w = np.abs(np.random.randn(x.shape[0], k))
    h = np.abs(np.random.randn(k, x.shape[1]))

    # print('Loading initial matrices.')
    # inimat = sio.loadmat('/home/ralf/uni/data/msot-matlab/k4_ini.mat')
    # w = inimat['W_ini']
    # h = inimat['H_ini']

    # precomputing w @ h
    # saves one computation each iteration
    wh = w @ h

    if kl:
        logging.info('Using Kullback-Leibler divergence.')
        obj_history = [dist_kl(x, wh)]
    else:
        logging.info('Using euclidean distance.')
        obj_history = [dist_euclid(x, wh)]


    print('Entering Main Loop.')
    # Main iteration
    for i in range(max_iter):
        old_obj = obj_history[-1]

        # Update step
        w = w_update(kl, x, w, h, wh, alpha_w, norm)
        h = h_update(kl, x, w, h, w @ h, alpha_h1, alpha_h2, norm)
        wh = w @ h

        # get new distance
        if kl:
            new_obj = dist_kl(x, wh)
        else:
            new_obj = dist_euclid(x, wh)


        # Iteration info
        logging.info('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # Check convergence; save and break iteration
        if convergence_check(new_obj, old_obj, tol1, tol2):
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            logging.warning('Results saved in {}.'.format(save_str))
            break

        # save every XX iterations
        if i % 100 == 0:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            logging.warning('Saved on iteration {} in {}.'.format(i, save_str))

        # save on max_iter
        if i == max_iter-1:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            logging.warning('Max iteration. Results saved in {}.'.format(save_str))


@begin.start
@begin.logging
def main(param_file='parameter_file', *, transpose=False):
    """ NMF with MUR """

    try:
        params = import_module(param_file)
    except ImportError:
        print('No parameter file found.')
        return

    try:
        if params.load_var == 'LOAD_MSOT':
            data = loadme.msot(params.load_file)
            logging.info('Loaded MSOT data.')
        else:
            data = loadme.mat(params.load_file, params.load_var)
            logging.info('Loaded PET data.')
    except AttributeError:
        print('No file/variable given.')
        return

    if data.ndim == 3:
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]), order='F')
        logging.info('Data was 3D. Reshaped to 2D.')

    if transpose:
        data = data.T

    try:
        mur(data,
            params.features,
            kl=params.kl,
            norm=params.norm,
            max_iter=params.max_iter,
            tol1=params.tol1,
            tol2=params.tol2,
            alpha_w=params.alpha_w,
            alpha_h1=params.alpha_h1,
            alpha_h2=params.alpha_h2,
            save_dir=params.save_dir,
            save_file=params.save_file,
            )
    except NameError as e:
        raise Exception('Parameter file incomplete.').with_traceback(e.__traceback__)
