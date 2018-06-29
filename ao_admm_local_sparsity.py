# system imports
import os
# noinspection PyUnresolvedReferences
import better_exceptions

# math imports
import numpy as np
from numpy.linalg import norm, solve
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# personal imports
from utils import convergence_check, distance, nndsvd, save_results
from misc import showme

import pdb


def initialize(data, features, loss):

    w, h = nndsvd(data, features)
    w = np.abs(np.random.randn(data.shape[0], features))
    h = np.abs(np.random.randn(features, data.shape[1]))
    w_aux = np.zeros_like(w)
    dual_w = np.zeros_like(w)
    dual_h = np.zeros_like(h)

    if loss == 'kl':
        y_dual = np.zeros_like(data)
    else:
        # y_dual = None
        y_dual = np.zeros_like(data)

    return w, h, w_aux, dual_w, dual_h, w@h, y_dual


def terminate(mat, mat_prev, aux, dual, tol=1e-2):

    # relative primal residual
    r = norm(mat - aux)/norm(mat)
    # relative dual residual
    s = norm(mat - mat_prev)/norm(dual)

    if r < tol and s < tol:
        return True
    else:
        return False


def admm_ls_update(y, w, h, dual, k, prox_type='nn', *, admm_iter=10, lambda_=0):
    """ ADMM update for NMF subproblem, when one of the factors is fixed

    using least-squares loss
    """

    # precompute all the things
    g = w.T @ w
    rho = np.trace(g)/k
    cho = la.cholesky(g + rho * np.eye(g.shape[0]), lower=True)
    wty = w.T @ y

    for i in range(admm_iter):
        h_aux = la.cho_solve((cho, True), wty + rho * (h + dual))
        h_prev = h.copy()
        h = prox(prox_type, h_aux.T, dual.T, rho=rho, lambda_=lambda_)
        dual = dual + h - h_aux

        if terminate(h, h_prev, h_aux, dual):
            print('ADMM break after {} iterations.'.format(i))
            break

    # showme.im1d(h.T)
    return h, dual


def admm_kl_update(v, v_aux, dual_v, w, h, dual_h, k, prox_type='nn',
                   *, admm_iter=10, lambda_=0):
    """ ADMM update for NMF subproblem, when one of the factors is fixed

    using Kullback-Leibler loss
    """

    # precompute all the things
    g = w.T @ w
    rho = np.trace(g)/k
    cho = la.cholesky(g + rho * np.eye(g.shape[0]), lower=True)

    for i in range(admm_iter):
        # h_aux  and h update
        h_aux = la.cho_solve((cho, True), w.T @ (v_aux + dual_v) + rho * (h + dual_h))
        h_prev = h.copy()
        h = prox(prox_type, h_aux.T, dual_h.T, rho=rho, lambda_=lambda_)

        # v_aux update
        v_bar = w @ h_aux - dual_v
        v_aux = 1/2 * ((v_bar-1) + np.sqrt((v_bar-1)**2 + 4*v))

        # dual variables updates
        dual_h = dual_h + h - h_aux
        dual_v = dual_v + v_aux - w @ h_aux

        if terminate(h, h_prev, h_aux, dual_h):
            print('ADMM break after {} iterations.'.format(i))
            break

    return h, dual_h, v_aux, dual_v


def admm_local_sparsity(v, v_aux, dual_v, w_aux, dual_w, h, k, admm_iter=20):
    g = h @ h.T
    rho1 = np.trace(g)/k
    rho2 = rho1

    eps = {'abs': la.norm(v),
           'rel': 1e-3}
    eps['pri1'] = np.sqrt(w_aux.shape[0]*w_aux.shape[1]) * eps['abs']
    eps['pri2'] = np.sqrt(w_aux.shape[0]*h.shape[1]) * eps['abs']
    eps['dual'] = eps['pri1']

    tau = {'incr1': 2, 'decr1': 2, 'incr2': 2, 'decr2': 2}
    eta1, eta2 = 1, 1

    for i in range(admm_iter):
        # H update
        a = rho1 * np.eye(g.shape[0]) + rho2 * g
        b = rho1 * (w_aux - dual_w) + rho2 * (v_aux - dual_v) @ h.T
        w = b @ np.linalg.inv(a)
        w = np.where(w < 0, 0, w)

        # H tilde update
        w_aux_old = w_aux.copy()
        w_aux = local_sparsity(w_aux, dual_w, lambda_=1, rho=rho1, upper_bound=1)

        # Y tilde update
        a = sp.eye(v.shape[0]) - rho2 * sp.eye(v.shape[0])
        b = v - rho2 * (w @ h + dual_v)
        v_aux_old = v_aux.copy()
        v_aux = spla.spsolve(a, b)

        # U and V update
        dual_w = dual_w - (w_aux - w)
        dual_v = dual_v - (v_aux - w@h)

        # update residuals
        eps, r1, r2, s = update_residuals(eps, rho1, rho2, v, v_aux, v_aux_old, dual_v, w,
                                          w_aux, w_aux_old, dual_w, h, tau, eta1, eta2)

        if la.norm(r1) >= eps['pri1'] and \
            la.norm(r2) >= eps['pri2'] and \
            la.norm(s) >= eps['dual']:
            break

    showme.im2d(w.reshape((257, 256, k), order='F'))

    return w, w_aux, dual_w, v_aux, dual_v


def local_sparsity(mat_aux, dual, lambda_, rho, upper_bound):
    mat = np.zeros_like(mat_aux)

    pos = mat_aux + dual - lambda_ / rho * np.ones_like(mat_aux)
    pos = np.where(pos < 0, 0, pos)

    for i in range(pos.shape[0]):
        if np.sum(pos[i, :]) <= upper_bound:
            mat[i, :] = pos[i, :]
        else:
            ones = np.ones_like(mat[i, :])

            val = -np.sort(-(mat_aux[i, :] - dual[i, :]))
            for j in range(1, mat_aux.shape[1]+1):
                test = rho * val[j-1] + lambda_ \
                       - rho/j * (np.sum(val[:j]) + lambda_/rho - upper_bound)
                if test < 0:
                    index_count = j-1
                    break
            else:
                index_count = mat_aux.shape[1] + 1

            theta = rho / index_count \
                    * (np.sum(val[:(index_count+1)]) + lambda_ / rho - upper_bound)
            shrink = mat_aux[i, :] + dual[i, :] - lambda_/rho * ones - theta / rho * ones
            mat[i, :] = np.where(shrink < 0, 0, shrink)

    return mat


def update_residuals(eps, rho1, rho2, v, v_aux, v_aux_old, dual_v, w, w_aux, w_aux_old,
                     dual_w, h, tau, eta1, eta2):

    # Residuals
    s = rho1 * (w_aux_old - w_aux) + rho2 * (v_aux_old - v_aux) @ h.T
    r1 = rho1 * (w_aux - w)
    r2 = rho2 * (v_aux - w@h)

    if la.norm(r1) > eta1 * la.norm(s):
        rho1 = rho1 * tau['incr1']
        dual_w = dual_w / tau['incr1']
    elif la.norm(s) > eta1 * la.norm(r1):
        rho1 = rho1 / tau['decr1']
        dual_w = dual_w * tau['decr1']

    if la.norm(r2) > eta2 * la.norm(s):
        rho2 = rho2 * tau['incr2']
        dual_v = dual_v / tau['incr2']
    elif la.norm(s) > eta2 * la.norm(r2):
        rho2 = rho2 / tau['decr2']
        dual_v = dual_v * tau['decr2']

    eps['pri1'] = np.sqrt(w.shape[0]*w.shape[1]) * eps['abs'] \
                  + eps['rel'] * max(la.norm(w), la.norm(w_aux), 0)
    eps['pri2'] = np.sqrt(w.shape[0]*h.shape[1]) * eps['abs'] \
                  + eps['rel'] * max(la.norm(w@h), la.norm(v_aux), 0)
    eps['dual'] = np.sqrt(w.shape[0]*w.shape[1]) * eps['abs'] \
                  + eps['rel'] * la.norm(rho1 * dual_w + rho2 * dual_v @ h.T)

    return eps, r1, r2, s


def prox(prox_type, mat_aux, dual, *, rho=None, lambda_=None, upper_bound=1):
    """ proximal operators for

    nn : non-negativity
    l1n : l1-norm with non-negativity
    l2n : l2-norm with non-negativity
    l1inf : l1,inf-norm
    """

    if prox_type == 'nn':
        diff = mat_aux - dual

        mat = np.where(diff < 0, 0, diff)
        return mat.T

    elif prox_type == 'l1n':
        diff = mat_aux - dual
        mat = diff - lambda_/rho

        mat = np.where(mat < 0, 0, mat)
        return mat.T

    elif prox_type == 'l2n':
        n = mat_aux.shape[0]
        k = -np.array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)])
        offset = [-1, 0, 1]
        tikh = sp.diags(k, offset)  # .toarray()

        # matinv = la.inv(lambda_ * tikh.T @ tikh + rho * np.eye(n))
        # mat = rho * matinv @ (mat_dual - dual)

        # a had 1/rho instead of rho?
        a = 1/rho * (lambda_ * tikh.T @ tikh + rho * sp.eye(n))
        b = mat_aux - dual
        mat = spla.spsolve(a, b)

        mat = np.where(mat < 0, 0, mat)
        return mat.T

    elif prox_type == 'l1inf':
        mat = np.zeros_like(mat_aux)

        pos = mat_aux + dual - lambda_ / rho * np.ones_like(mat_aux)
        pos = np.where(pos < 0, 0, pos)

        for i in range(pos.shape[0]):
            if np.sum(pos[i, :]) <= upper_bound:
                mat[i, :] = pos[i, :]
            else:
                ones = np.ones_like(mat[i, :])

                val = -np.sort(-(mat_aux[i, :] - dual[i, :]))
                for j in range(1, mat_aux.shape[1]+1):
                    test = rho * val[j-1] + lambda_ \
                           - rho/j * (np.sum(val[:j]) + lambda_/rho - upper_bound)
                    if test < 0:
                        index_count = j-1
                        break
                else:
                    index_count = mat_aux.shape[1] + 1

                theta = rho / index_count \
                        * (np.sum(val[:(index_count+1)]) + lambda_ / rho - upper_bound)
                shrink = mat_aux[i, :] + dual[i, :] - lambda_ / rho * ones \
                         - theta / rho * ones
                mat[i, :] = np.where(shrink < 0, 0, shrink)

        return mat.T

    elif prox_type == 'l1inf_transpose':
        mat = np.zeros_like(mat_aux)

        pos = mat_aux + dual - lambda_ / rho * np.ones_like(mat_aux)
        pos = np.where(pos < 0, 0, pos)
        print('will go {}'.format(pos.shape[1]))
        for i in range(pos.shape[1]):
            if np.sum(pos[:, i]) <= upper_bound:
                mat[:, i] = pos[:, i]
            else:
                ones = np.ones_like(mat[:, i])
                val = mat_aux[:, i] - dual[:, 1]
                val = -np.sort(-val)
                for j in range(1, mat_aux.shape[0]+1):
                    test = rho * val[j-1] + lambda_ - rho/j * (np.sum(val[:j]) + lambda_/rho - upper_bound)
                    if test < 0:
                        index_count = j-1
                        break
                else:
                    index_count = mat_aux.shape[0] + 1
                theta = rho / index_count * (np.sum(val[:(index_count+1)]) + lambda_ / rho - upper_bound)
                theta = theta if theta > 0 else 0
                shrink = mat_aux[:, i] + dual[:, i] - lambda_ / rho * ones - theta / rho * ones
                mat[:, i] = np.where(shrink < 0, 0, shrink)

        return mat.T

    else:
        raise TypeError('Unknown prox_type.')


def ao_admm(v, k, *, distance_type='eu', loss_type='ls', reg_w=(0, 'nn'),
            reg_h=(0, 'l2n'), min_iter=10, max_iter=100000, admm_iter=10,
            tol1=1e-3, tol2=1e-3, save_dir='./results/'):
    """ AO-ADMM framework for NMF

    following paper by:
    Huang, Sidiropoulos, Liavas (2015)
    A flexible and efficient algorithmic framework for constrained matrix and tensor
    factorization
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_local_{feat}_{dist}_{loss}_{lambda_w}:{prox_w}_{lambda_h}:{prox_h}'.format(
        feat=k,
        dist=distance_type,
        loss=loss_type,
        lambda_w=reg_w[0],
        prox_w=reg_w[1],
        lambda_h=reg_h[0],
        prox_h=reg_h[1],
    )
    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'min_iter': min_iter,
                       'max_iter': max_iter,
                       'admm_iter': admm_iter,
                       'tol1': tol1,
                       'tol2': tol2,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # initialize
    w, h, w_aux, dual_w, dual_h, v_aux, dual_v = initialize(v, k, loss_type)

    # initial distance value
    obj_history = [distance(v, w@h, distance_type=distance_type)]

    # Main iteration
    for i in range(max_iter):
        if loss_type == 'ls':
            h, dual_h = admm_ls_update(v, w, h, dual_h, k,
                                       lambda_=reg_h[0],
                                       prox_type=reg_h[1],
                                       admm_iter=admm_iter)
            w, w_aux, dual_w, v_aux, dual_v = admm_local_sparsity(v, v_aux, dual_v,
                                                                  w_aux, dual_w, h, k,
                                                                  admm_iter=admm_iter)

        elif loss_type == 'kl':
            h, dual_h, v_aux, dual_v = admm_kl_update(v, v_aux, dual_v, w, h, dual_h, k,
                                                      lambda_=reg_h[0],
                                                      prox_type=reg_h[1],
                                                      admm_iter=admm_iter)
            w, w_aux, dual_w, v_aux, dual_v = admm_local_sparsity(v, v_aux, dual_v,
                                                                  w_aux, dual_w, h, k,
                                                                  admm_iter=admm_iter)

        else:
            raise TypeError('Unknown loss function type.')

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
        if i % 1 == 0:
            save_results(save_str, w, h, i, obj_history, experiment_dict)

    else:
        # save on max_iter
        save_results(save_str, w, h, max_iter, obj_history, experiment_dict)
        print('Max iteration reached.')
