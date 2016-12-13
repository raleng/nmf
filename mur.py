#!/usr/bin/python3
import begin
import logging
import numpy as np
import os
from importlib import import_module
from misc import loadme
from os.path import isfile, join


def dist_euclid(x, wh):
    """Euclidean distance"""
    value = 0.5 * np.sum((x - wh) ** 2)
    return value


def dist_kl(x, wh):
    """Kullback-Leibler divergence"""
    value = x * np.log(x / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - x + wh)
    return value


def wh_update_euclid(x, w, h, wh, alpha_w, alpha_h):
    """MUR Update with euclidean distance"""
    h = h * (w.T @ x) / (w.T @ wh + alpha_h * h + 1e-9)
    w = w * (x @ h.T) / (w @ (h @ h.T) + alpha_w * w + 1e-9)
    return w, h


def wh_update_kl(x, w, h, wh):
    """MUR Update with Kullback-Leibler divergence"""
    h = h * ((w.T @ (x / (wh + 1e-9))) / np.sum(w, 0)[:, np.newaxis])
    w = w * (h @ (x / (wh + 1e-9)).T / np.sum(h, 1)[:, np.newaxis]).T
    return w, h


def mur(x, k, *, kl=False, max_iter=100000, tol1=1e-3, tol2=1e-3, alpha_w=0.0, alpha_h=0.0,
        save_dir="./results/", save_file="nmf"):
    """ NMF with MUR

    Expects following arguments:
    X -- 2D Data
    k -- number of components

    Accepts keyword arguments:
    kl -- BOOL: if True, use Kullback Leibler, else Euclidean
    maxiter -- INT: maximum number of iterations
    tol1 -- FLOAT: convergence tolerance
    tol2 -- FLOAT: convergence tolerance
    alpha_W -- FLOAT: regularization parameter for w-Update
    alpha_H -- FLOAT: regularization parameter for h-Update
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    experiment_info = """
                        k: {k}
                        kl: {kl}
                        max_iter: {max_iter}
                        tol1: {tol1}
                        tol2: {tol2}
                        alpha_w: {alpha_w}
                        alpha_h: {alpha_h}
                      """.format(
                              k=k,
                              kl=kl,
                              max_iter=max_iter,
                              tol1=tol1,
                              tol2=tol2,
                              alpha_w=alpha_w,
                              alpha_h=alpha_h,
                              )

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = len(str(tol)) if tol < 1 else 0

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)

    save_str = '{file_path}_{k}_{dist}'.format(file_path=join(save_dir, save_file),
                                               k=k,
                                               dist=('KL' if kl else 'EU'),
                                               )

    # make sure data is positive; should be anyways but data could contain small
    # negative numbers due to rounding errors and such
    if np.min(x) < 0:
        x += abs(np.min(x))
        logging.info('Data elevated.')

    # normalizing
    x /= np.max(x[:])

    # initializing w and h with random matrices
    w = np.abs(np.random.randn(x.shape[0], k))
    h = np.abs(np.random.randn(k, x.shape[1]))

    # print('Loading initial matrices.')
    # inimat = sio.loadmat('/home/ralf/uni/data/msot-matlab/k4_ini.mat')
    # w = inimat['W_ini']
    # h = inimat['H_ini']

    # w @ h is needed in several places
    wh = (w @ h)

    if kl:
        logging.info('Using Kullback-Leibler divergence.')
        obj_history = [dist_kl(x, wh)]
    else:
        logging.info('Using euclidean distance.')
        obj_history = [dist_euclid(x, wh)]

    for i in range(max_iter):
        old_obj = obj_history[-1]

        if i == max_iter-1:
            np.savez(save_str, w=w, h=h, i=i, objhistory=obj_history,
                    experiment_info=experiment_info)
            logging.warning('Max iteration. Results saved in {}'.format(save_str))

        # Update step
        if kl:
            w, h = wh_update_kl(x, w, h, wh)
        else:
            w, h = wh_update_euclid(x, w, h, wh, alpha_w, alpha_h)

        # normalization
        norms = np.sqrt(np.sum(h.T**2, 0))
        h = h / norms[:, None]
        w = w * norms

        wh = w @ h

        # get new distance
        if kl:
            new_obj = dist_kl(x, wh)
        else:
            new_obj = dist_euclid(x, wh)

        # Iteration info
        logging.info('[{}]: {:.{}f}'.format(i, new_obj, tol_precision))
        obj_history.append(new_obj)

        # Check convergence
        break_true = True
        if new_obj < tol1:
            logging.warning('Algorithm converged (1)')
        elif new_obj >= old_obj-tol2:
            logging.warning('Algorithm converged (2)')
        else:
            pass

        break_true = False

        if break_true:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                    experiment_info=experiment_info)
            logging.warning('Results saved in {}'.format(save_str))
            break

        # save every XX iterations
        if i % 100 == 0:
            np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
                    experiment_info=experiment_info)
            logging.warning('Saved on iteration {} in {}'.format(i, save_str))


@begin.start
@begin.logging
def main(param_file='parameter_file'):
    """ NMF with MUR """

    try:
        params = import_module(param_file)
    except:
        print('No parameter file found.')
        return

    try:
        if params.load_var == 'LOAD_MSOT':
            data = loadme.msot(params.load_file)
            logging.info('Loaded MSOT data.')
        else:
            data = loadme.pet(params.load_file, params.load_var)
            logging.info('Loaded PET data.')
    except AttributeError:
        print('No file/variable given.')
        return

    if data.ndim == 3:
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
        logging.info('Data was 3D. Reshaped to 2D.')

    try:
        mur(data,
            params.features,
            kl=params.kl,
            max_iter=params.max_iter,
            tol1=params.tol1,
            tol2=params.tol2,
            alpha_w=params.alpha_w,
            alpha_h=params.alpha_h,
            save_dir=params.save_dir,
            save_file=params.save_file,
            )
    except NameError:
        print('Parameter file incomplete.\n')
        raise
