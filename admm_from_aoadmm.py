# system imports
import os
# noinspection PyUnresolvedReferences
import better_exceptions

# math imports
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# personal imports
from utils import convergence_check, distance, nndsvd, save_results


def initialize(data, features, loss):

    w, h = nndsvd(data, features, variant='zero')
    # w = np.abs(np.random.randn(data.shape[0], features))
    # h = np.abs(np.random.randn(features, data.shape[1]))
    w_aux = w.copy()
    h_aux = h.copy()
    dual_w = np.zeros_like(w)
    dual_h = np.zeros_like(h)

    y_dual = np.zeros_like(data)

    return w, h, w_aux, h_aux, dual_w, dual_h, y_dual, y_dual


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
        print(np.max(h))
        dual = dual + h - h_aux

        if terminate(h, h_prev, h_aux, dual):
            print('ADMM break after {} iterations.'.format(i))
            break

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
        return mat

    elif prox_type == 'l1n':
        diff = mat_aux - dual
        mat = diff - lambda_/rho

        mat = np.where(mat < 0, 0, mat)
        return mat

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
        return mat

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
                    test = rho * val[j-1] + lambda_ - rho/j * (np.sum(val[:j]) + lambda_/rho - upper_bound)
                    if test < 0:
                        index_count = j-1
                        break
                else:
                    index_count = mat_aux.shape[1] + 1

                theta = rho / index_count * (np.sum(val[:(index_count+1)]) + lambda_ / rho - upper_bound)
                shrink = mat_aux[i, :] + dual[i, :] - lambda_ / rho * ones - theta / rho * ones
                mat[i, :] = np.where(shrink < 0, 0, shrink)

        return mat

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

        return mat

    else:
        raise TypeError('Unknown prox_type.')


def aux_update(mat, dual, other_aux, data_aux, data_dual, rho):
    a = other_aux.T @ other_aux + rho * np.eye(other_aux.shape[1])
    b = other_aux.T @ (data_aux + data_dual) + rho * (mat + dual)
    mat_aux = np.linalg.solve(a, b)
    return mat_aux


def admm(v, k, *, rho=1, distance_type='eu', loss_type='ls', reg_w=(0, 'nn'), reg_h=(0, 'l2n'),
         min_iter=10, max_iter=100000, admm_iter=10, tol1=1e-3, tol2=1e-3, save_dir='./results/'):
    """ AO-ADMM framework for NMF

    following paper by:
    Huang, Sidiropoulos, Liavas (2015)
    A flexible and efficient algorithmic framework for constrained matrix and tensor
    factorization
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_admmfromaoadmm_{feat}_{rho}_{dist}_{loss}_{lambda_w}:{prox_w}_{lambda_h}:{prox_h}'.format(
        feat=k,
        rho=rho,
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
    w, h, w_aux, h_aux, dual_w, dual_h, v_aux, dual_v = initialize(v, k, loss_type)

    # initial distance value
    obj_history = [distance(v, w@h, distance_type=distance_type)]

    # Main iteration
    for i in range(max_iter):
        h_aux = aux_update(h, dual_h, w_aux, v_aux, dual_v, rho)
        w_aux = aux_update(w.T, dual_w.T, h_aux.T, v_aux.T, dual_v.T, rho)
        w_aux = w_aux.T

        h = prox(reg_h[1], h_aux, dual_h, rho=rho, lambda_=reg_h[0])
        w = prox(reg_w[1], w_aux.T, dual_w.T, rho=rho, lambda_=reg_w[0])
        w = w.T

        v_bar = w_aux @ h_aux - dual_v
        v_aux = 1/2 * ((v_bar-1) + np.sqrt((v_bar-1)**2 + 4*v))

        dual_h = dual_h + h - h_aux
        dual_w = dual_w + w - w_aux
        dual_v = dual_v + v_aux - w_aux @ h_aux

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
