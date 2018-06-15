# system imports
import os
# noinspection PyUnresolvedReferences
import better_exceptions

# math imports
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# personal imports
from utils import convergence_check, distance, nndsvd, save_results


def initialize(data, features):
    w, h = nndsvd(data, features)
    dual_w = np.zeros_like(w)
    dual_h = np.zeros_like(h)
    return w, h, dual_w, dual_h


def admm_update(y, w, h, dual, k, prox_type='nn', *, admm_iter=10, lambda_=0):
    # init h and dual
    g = w.T @ w
    rho = np.trace(g)/k
    cho = la.cholesky(g + rho * np.eye(g.shape[0]), lower=True)
    wty = w.T @ y

    for i in range(admm_iter):
        h_dual = la.cho_solve((cho, True), wty + rho * (h + dual))
        h = prox(prox_type, h_dual.T, dual.T, rho=rho, lambda_=lambda_)
        dual = dual + h - h_dual
    return h, dual


def prox(prox_type, mat_dual, dual, *, rho=None, lambda_=None):

    if prox_type == 'l2n':
        n = mat_dual.shape[0]
        k = -np.array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)])
        offset = [-1, 0, 1]
        tikh = sp.diags(k, offset) # .toarray()

        # matinv = la.inv(lambda_ * tikh.T @ tikh + rho * np.eye(n))
        # mat = rho * matinv @ (mat_dual - dual)

        a = 1/rho * (lambda_ * tikh.T @ tikh + rho * sp.eye(n))
        b = mat_dual - dual
        mat = spla.spsolve(a, b)

        mat = (mat >= 0) * mat
        return mat.T

    elif prox_type == 'nn':
        diff = mat_dual - dual
        mat = (diff >= 0) * diff
        return mat.T

    else:
        raise TypeError('Unknown prox_type.')


def ao_admm(v, k, *, distance_type='eu', reg_w=(0, 'nn'), reg_h=(0, 'l2n'), min_iter=10,
            max_iter=100000, admm_iter=10, tol1=1e-3, tol2=1e-3, save_dir='./results/'):

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_aoadmm_{feat}_{dist}_{lambda_w}:{prox_w}_{lambda_h}:{prox_h}'.format(
        feat=k,
        dist=distance_type,
        lambda_w=reg_w[0],
        prox_w=reg_w[1],
        lambda_h=reg_h[0],
        prox_h=reg_h[1],
    )
    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'max_iter': max_iter,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # initialize
    w, h, dual_w, dual_h = initialize(v, k)

    # initial distance value
    obj_history = [distance(v, w@h, distance_type=distance_type)]

    # Main iteration
    for i in range(max_iter):
        h, dual_h = admm_update(v, w, h, dual_h, k,
                                lambda_=reg_h[0],
                                prox_type=reg_h[1],
                                admm_iter=admm_iter)
        w, dual_w = admm_update(v.T, h.T, w.T, dual_w.T, k,
                                lambda_=reg_w[0],
                                prox_type=reg_w[1],
                                admm_iter=admm_iter)
        w = w.T
        dual_w = dual_w.T

        # Iteration info
        obj_history.append(distance(v, w@h, distance_type=distance_type))
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