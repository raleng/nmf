import os
from importlib import import_module

import begin
# noinspection PyUnresolvedReferences
# import better_exceptions
import numpy as np
from math import sqrt
from misc import loadme
from scipy import optimize
from sklearn.preprocessing import normalize

import fcnnls


def distance(x, wh):
    """ Kullback-Leibler divergence """
    # value = 0.5 * np.sum((x - wh) ** 2)
    value = x * np.log(x / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - x + wh)
    return value


def w_update(x, h, lambda_w, *, use_fcnnls=False):
    """ Update W """

    a = np.concatenate((h.T, sqrt(2*lambda_w) * np.eye(h.shape[0])))
    b = np.concatenate((x.T, np.zeros((h.shape[0], x.shape[0]))))

    if use_fcnnls:
        w = fcnnls.fcnnls(a, b)
    else:
        w = np.zeros((a.shape[1], b.shape[1]))
        for i in range(b.shape[1]):
            w[:, i], _ = optimize.nnls(a, b[:, i])

    return w.T


def h_update(x, w, lambda_h, *, use_fcnnls=False):
    """ Update H """

    a = np.concatenate((w, sqrt(2*lambda_h) * np.eye(w.shape[1])))
    b = np.concatenate((x, np.zeros((w.shape[1], x.shape[1]))))

    if use_fcnnls:
        h = fcnnls.fcnnls(a, b)
    else:
        h = np.zeros((a.shape[1], b.shape[1]))
        for i in range(b.shape[1]):
            h[:, i], _ = optimize.nnls(a, b[:, 1])

    return h


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


def save_results(convergence, max_iter, i, save_str, w, h, obj_history, experiment_dict):
    # Check convergence; save and break iteration
    if convergence and i > 10:
        np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                 experiment_dict=experiment_dict)
        print('Results saved in {}.'.format(save_str))
        return True

    # save every XX iterations
    if i % 100 == 0:
        np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                 experiment_dict=experiment_dict)
        print('Saved on iteration {} in {}.'.format(i, save_str))
        return False

    # save on max_iter
    if i == max_iter-1:
        np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                 experiment_dict=experiment_dict)
        print('Max iteration. Results saved in {}.'.format(save_str))
        return False


def anls(x, k, *, use_fcnnls=False, lambda_w=0, lambda_h=0, max_iter=1000, tol1=1e-3, tol2=1e-3,
         save_dir='./results/', save_file='nmf_anls'):
    """ NMF via ANLS with FCNNLS """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_str = os.path.join(save_dir, save_file)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'max_iter': max_iter,
                       'lambda_w': lambda_w,
                       'lambda_h': lambda_h,
                       'tol1': tol1,
                       'tol2': tol2
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    h = np.abs(np.random.randn(k, x.shape[1]))
    h, norm = normalize(h, return_norm=True)

    obj_history = [1e10]

    for i in range(max_iter):

        # Update step
        w = w_update(x, h, lambda_w, use_fcnnls=use_fcnnls)
        w = w * norm

        h = h_update(x, w, lambda_h, use_fcnnls=use_fcnnls)
        h, norm = normalize(h, return_norm=True)

        # w, norm = normalize(w, axis=0, return_norm=True)
        # h = (h.T * norm).T

        new_obj = distance(x, w@h)

        # Iteration info
        print('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # check convergence and save
        converged = convergence_check(new_obj, obj_history[-2], tol1, tol2)
        if save_results(converged, max_iter, i, save_str, w, h, obj_history, experiment_dict):
            break


@begin.start
def main(param_file='parameters_anls'):
    """ NMF with ANLS """

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

    anls(data,
         params.features,
         use_fcnnls=params.use_fcnnls,
         lambda_w=params.lambda_w,
         lambda_h=params.lambda_h,
         max_iter=params.max_iter,
         tol1=params.tol1,
         tol2=params.tol2,
         save_dir=params.save_dir,
         save_file=params.save_file,
         )
