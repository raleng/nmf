#!/usr/bin/env python3
import os
from importlib import import_module

import begin
# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np
from misc import loadme


def initialize(dims, features):
    """ Initializing variables """

    w = np.abs(np.random.randn(dims[0], features))
    h = np.abs(np.random.randn(features, dims[1]))
    x = w @ h

    w_p = w.copy()
    h_p = h.copy()

    alpha_x = np.zeros(x.shape)
    alpha_w = np.zeros(w.shape)
    alpha_h = np.zeros(h.shape)

    return x, w, h, w_p, h_p, alpha_x, alpha_w, alpha_h


def distance(v, wh):
    """ Kullback-Leibler divergence """
    value = v * np.log(v / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - v + wh)
    return value


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


def x_update(v, wh, alpha_x, rho):
    """ ADMM update of X """
    value = rho * wh - alpha_x - 1
    x = value + np.sqrt(value**2 + 4 * rho * v)
    x /= 2 * rho
    return x


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


def convergence_check(new, old, tol1, tol2):
    """ Checks the convergence criteria """

    convergence_break = True

    if new < tol1:
        print('Algorithm converged (1).')
    elif new >= old - tol2:
        print('Algorithm converged (2).')
    else:
        convergence_break = False

    return convergence_break


def admm(v, k, *, rho=1, max_iter=100000, tol1=1e-3, tol2=1e-3, save_dir='./results/', save_file='nmf_admm'):
    """ NMF with ADMM

    Expects following arguments:
    v -- 2D data
    k -- number of components

    Accepts keyword arguments:
    max_iter -- INT: maximum number of iterations (default: 100000)
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_str = os.path.join(save_dir, save_file)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'max_iter': max_iter,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    x, w, h, w_p, h_p, alpha_x, alpha_w, alpha_h = initialize(v.shape, k)

    # initial distance value
    obj_history = [distance(v, w@h)]

    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(x, h, w_p, alpha_x, alpha_w, rho)
        h = h_update(x, w, h_p, alpha_x, alpha_h, rho)
        wh = w @ h

        x = x_update(v, wh, alpha_x, rho)

        w_p, h_p = wh_p_update(w, h, alpha_w, alpha_h, rho)
        alpha_x, alpha_h, alpha_w, = alpha_update(x, w, h, wh, w_p, h_p, alpha_x, alpha_w, alpha_h, rho)

        # get new distance
        new_obj = distance(v, w_p@h_p)

        # Iteration info
        print('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # Check convergence; save and break iteration
        if i > 10 and convergence_check(new_obj, obj_history[-2], tol1, tol2):
            np.savez(save_str, w=w, h=h, w_p=w_p, h_p=h_p, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Results saved in {}.'.format(save_str))
            break

        # save every XX iterations
        if i % 100 == 0:
            np.savez(save_str, w=w, h=h, w_p=w_p, h_p=h_p, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Saved on iteration {} in {}.'.format(i, save_str))

    else:
        np.savez(save_str, w=w, h=h, w_p=w_p, h_p=h_p, i=max_iter, obj_history=obj_history,
                 experiment_dict=experiment_dict)
        print('Max iteration. Results saved in {}.'.format(save_str))


@begin.start
def main(param_file='parameters_admm'):
    """ NMF with ADMM """

    try:
        params = import_module(param_file)
    except ImportError:
        print('No parameter file found.')
        return

    try:
        if params.load_var == 'LOAD_MSOT':
            data = loadme.msot(params.load_file)
            print('Loaded MSOT data.')
        else:
            data = loadme.mat(params.load_file, params.load_var)
            print('Loaded PET data.')
    except AttributeError:
        print('No file/variable given.')
        return

    if data.ndim == 3:
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]), order='F')
        print('Data was 3D. Reshaped to 2D.')

    try:
        admm(data,
             params.features,
             rho=params.rho,
             max_iter=params.max_iter,
             tol1=params.tol1,
             tol2=params.tol2,
             save_dir=params.save_dir,
             save_file=params.save_file,
             )
    except NameError as e:
        raise Exception('Parameter file incomplete.').with_traceback(e.__traceback__)
