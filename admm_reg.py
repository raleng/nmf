#!/usr/bin/env python3
import os
from importlib import import_module

import begin
# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np
from math import sqrt
from scipy import optimize

import bpp
from misc import loadme

# TODO scaling! normalize W, H accordingly

# TODO separate ANLS and ADMM


def initialize(dims, features):
    """ Initializing variables """
    w = np.abs(np.random.randn(dims[0], features))
    h = np.abs(np.random.randn(features, dims[1]))
    x = w @ h
    alpha_x = np.zeros(x.shape)
    return x, w, h, alpha_x


def distance(v, wh):
    """ Kullback-Leibler divergence """
    value = v * np.log(v / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - v + wh)
    return value


def w_update(x, h, alpha_x, lambda_w, rho, *, use_bpp=False):
    """ ADMM update of W """

    mu = 1/rho * alpha_x
    a = np.concatenate((sqrt(rho/2) * h.T, sqrt(lambda_w) * np.eye(h.shape[0])))
    b = np.concatenate((sqrt(rho/2) * (x + mu).T, np.zeros((h.shape[0], x.shape[0]))))
    # A = np.concatenate(h.T, sqrt(2*lambda_w) * np.eye(h.shape[0]))
    # b = np.concatenate(x.T, np.zeros((h.shape[0], x.shape[0])))

    if use_bpp:
        w = bpp.bpp(a, b)
    else:
        w = np.zeros((a.shape[1], b.shape[1]))
        for i in range(b.shape[1]):
            w[:, i], _ = optimize.nnls(a, b[:, i])

    return w.T


def h_update(x, w, alpha_x, lambda_h, rho, *, use_bpp=False):
    """ ADMM update of H """

    mu = 1/rho * alpha_x
    a = np.concatenate((sqrt(rho/2) * w, sqrt(lambda_h) * np.ones((1, w.shape[1]))))
    b = np.concatenate((sqrt(rho/2) * (x + mu), np.zeros((1, x.shape[1]))))
    # A = np.concatenate(w, sqrt(2*lambda_h) * np.eye(w.shape[1]))
    # b = np.concatenate(x, np.zeros(w.shape[1], x.shape[1]))

    if use_bpp:
        h = bpp.bpp(a, b)
    else:
        h = np.zeros((a.shape[1], b.shape[1]))
        for i in range(b.shape[1]):
            h[:, i], _ = optimize.nnls(a, b[:, 1])

    return h


def x_update(v, wh, alpha_x, rho):
    """ ADMM update of X """
    value = rho * wh - alpha_x - 1
    x = value + np.sqrt(value**2 + 4 * rho * v)
    x /= 2 * rho
    return x


def alpha_update(x, wh, alpha_x, rho):
    """ ADMM update dual variables """
    alpha_x = alpha_x + rho * (x - wh)
    return alpha_x


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


def admm(v, k, *, rho=1, use_bpp=False, lambda_w=0, lambda_h=0, max_iter=100000, tol1=1e-3, tol2=1e-3,
         save_dir='./results/', save_file='nmf_admm'):
    """ NMF with ADMM

    Expects following arguments:
    v -- 2D data
    k -- number of components

    Accepts keyword arguments:
    rho -- FLOAT: ADMM parameter
    lambda_w -- FLOAT: regularization parameter for W
    lambda_h -- FLOAT: regularization parameter for H
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
                       'rho': rho,
                       'lambda_w': lambda_w,
                       'lambda_h': lambda_h
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    x, w, h, alpha_x = initialize(v.shape, k)

    # initial distance value
    obj_history = [distance(v, w@h)]

    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(x, h, alpha_x, lambda_w, rho, use_bpp=use_bpp)
        h = h_update(x, w, alpha_x, lambda_h, rho, use_bpp=use_bpp)
        wh = w @ h

        x = x_update(v, wh, alpha_x, rho)

        alpha_x = alpha_update(x, wh, alpha_x, rho)

        # get new distance
        new_obj = distance(v, w@h)

        # Iteration info
        print('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # Check convergence; save and break iteration
        if i > 10 and convergence_check(new_obj, obj_history[-2], tol1, tol2):
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Results saved in {}.'.format(save_str))
            break

        # save every XX iterations
        if i % 100 == 0:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Saved on iteration {} in {}.'.format(i, save_str))

    else:
        # save on max_iter
        np.savez(save_str, w=w, h=h, i=max_iter, obj_history=obj_history,
                 experiment_dict=experiment_dict)
        print('Max iteration. Results saved in {}.'.format(save_str))


@begin.start
def main(param_file='parameters_admm_reg'):
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

    admm(data,
         params.features,
         rho=params.rho,
         bpp=params.bpp,
         lambda_w=params.lambda_w,
         lambda_h=params.lambda_h,
         max_iter=params.max_iter,
         tol1=params.tol1,
         tol2=params.tol2,
         save_dir=params.save_dir,
         save_file=params.save_file,
         )
