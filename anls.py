import os
from importlib import import_module

import begin
# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np
from misc import loadme
from sklearn.preprocessing import normalize

import fcnnls


def distance(x, wh):
    """ Kullback-Leibler divergence """
    value = x * np.log(x / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - x + wh)
    return value


def w_update(x, h, lambda_w):
    a = np.concatenate(h.T, sqrt(2*lambda_w) * np.eye(h.shape[0]))
    b = np.concatenate(x.T, np.zeros((h.shape[0], x.shape[0])))

    w = fcnnls.fcnnls(a, b)
    return w.T


def h_update(x, w, lambda_h):
    a = np.concatenate(w, sqrt(2*lambda_h) * np.eye(w.shape[1]))
    b = np.concatenate(x, np.zeros(w.shape[1], x.shape[1]))

    h = fcnnls.fcnnls(a, b)
    return h


def normalize(w, h):
    
    return w, h


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


def anls(x, features, *, lambda_w=0, lambda_h=0, max_iter=1000, tol1=1e-3, tol2=1e-3, save_dir='./results/', save_file='nmf_anls'):
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

    h = np.zeros(features, x.shape[1])

    obj_history = [np.inf]

    for i in range(max_iter):

        # Update step
        w = w_update(x, h, lambda_w)
        h = h_update(x, w, lambda_h)

        w, norm = normalize(w, return_norm=True)
        h = (h.T * norm).

        new_obj = distance(x, w@h)

        # Iteration info
        print('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # Check convergence; save and break iteration
        if convergence_check(new_obj, obj_history[-2], tol1, tol2) and i > 10:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Results saved in {}.'.format(save_str))
            break

        # save every XX iterations
        if i % 100 == 0:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Saved on iteration {} in {}.'.format(i, save_str))

        # save on max_iter
        if i == max_iter-1:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                     experiment_dict=experiment_dict)
            print('Max iteration. Results saved in {}.'.format(save_str))


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
         lambda_w=params.lambda_w,
         lambda_h=params.lambda_h,
         max_iter=params.max_iter,
         tol1=params.tol1,
         tol2=params.tol2,
         save_dir=params.save_dir,
         save_file=params.save_file,
         )
