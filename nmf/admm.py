# system imports
from collections import namedtuple
import logging
import os

# math imports
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# personal imports
from .utils import convergence_check, distance, nndsvd, save_results


def initialize(data, features, nndsvd_init):
    """ initialize variables for ADMM """
    # init w, h
    if nndsvd_init[0]:
        w, h = nndsvd(data, features, variant=nndsvd_init[1])
    else:
        w = np.abs(np.random.randn(data.shape[0], features))
        h = np.abs(np.random.randn(features, data.shape[1]))

    # init auxiliary variables for w, h
    w_aux = w.copy()
    h_aux = h.copy()

    # init dual variables for w, h, y
    dual_w = np.zeros_like(w)
    dual_h = np.zeros_like(h)
    y_dual = np.zeros_like(data)

    return w, h, w_aux, h_aux, dual_w, dual_h, y_dual, y_dual


def terminate(mat, mat_prev, aux, dual, tol=1e-2):
    """ Stops ADMM iteration of the subproblems according to primal/dual residual """ 

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

    # start iteration
    for i in range(admm_iter):
        # auxiliary update
        h_aux = la.cho_solve((cho, True), wty + rho * (h + dual))
        h_prev = h.copy()
        # h update
        h = prox(prox_type, h_aux.T, dual.T, rho=rho, lambda_=lambda_)
        # dual update
        dual = dual + h - h_aux

        # check residuals
        if terminate(h, h_prev, h_aux, dual):
            logging.info('ADMM break after {} iterations.'.format(i))
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

    # start iteration
    for i in range(admm_iter):
        # h_aux and h update
        h_aux = la.cho_solve((cho, True), w.T @ (v_aux + dual_v) + rho * (h + dual_h))
        h_prev = h.copy()
        h = prox(prox_type, h_aux.T, dual_h.T, rho=rho, lambda_=lambda_)

        # v_aux update
        v_bar = w @ h_aux - dual_v
        v_aux = 1/2 * ((v_bar-1) + np.sqrt((v_bar-1)**2 + 4*v))

        # dual variables updates
        dual_h = dual_h + h - h_aux
        dual_v = dual_v + v_aux - w @ h_aux

        # check residual
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

        # project negative values to zero
        mat = np.where(diff < 0, 0, diff)
        return mat

    elif prox_type == 'l1n':
        diff = mat_aux - dual
        mat = diff - lambda_/rho

        # project negative values to zero
        mat = np.where(mat < 0, 0, mat)
        return mat

    elif prox_type == 'l2n':
        n = mat_aux.shape[0]
        k = -np.array([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)])
        offset = [-1, 0, 1]
        tikh = sp.diags(k, offset)  # .toarray()

        # matinv = la.inv(lambda_ * tikh.T @ tikh + rho * np.eye(n))
        # mat = rho * matinv @ (mat_dual - dual)

        a = 1/rho * (lambda_ * tikh.T @ tikh + rho * sp.eye(n))
        b = mat_aux - dual
        mat = spla.spsolve(a, b)

        # project negative values to zero
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


def aux_update(mat, dual, other_aux, data_aux, data_dual, rho, distance_type):
    """ update the auxiliary admm variables """

    if distance_type == 'eu':
        a = other_aux.T @ other_aux + rho * np.eye(other_aux.shape[1])
        b = other_aux.T @ data_aux + rho * (mat + dual)

    elif distance_type == 'kl':
        a = other_aux.T @ other_aux + rho * np.eye(other_aux.shape[1])
        b = other_aux.T @ (data_aux + data_dual) + rho * (mat + dual)

    else:
        raise TypeError('Unknown loss type.')

    return np.linalg.solve(a, b)


def admm(v, k, *, rho=1, distance_type='eu', reg_w=(0, 'nn'), reg_h=(0, 'l2n'),
         min_iter=10, max_iter=100000, tol1=1e-3, tol2=1e-3, nndsvd_init=(True, 'zero'),
         save_dir='./results/'):
    """ AO-ADMM framework for NMF

    Following paper:
    Huang, Sidiropoulos, Liavas (2015)
    A flexible and efficient algorithmic framework for constrained matrix and tensor
    factorization


    Expects following arguments:
    x -- 2D Data
    k -- number of components

    Accepts keyword arguments:
    rho -- FLOAT: admm dampening parameter
    distance_type -- STRING: 'eu' for Euclidean, 'kl' for Kullback-Leibler
    reg_w -- Tuple(FLOAT, STRING): value und type of w-regularization
    reg_h -- Tuple(FLOAT, STRING): value und type of h-regularization
    min_iter -- INT: minimum number of iterations
    max_iter -- INT: maximum number of iterations
    tol1 -- FLOAT: convergence tolerance
    tol2 -- FLOAT: convergence tolerance
    nndsvd_init -- Tuple(BOOL, STRING): if BOOL = True, use NNDSVD-type STRING
    save_dir -- STRING: folder to which to save
    """

    # experiment parameters and results namedtuple
    Experiment = namedtuple('Experiment', 'method components rho distance_type nndsvd_init min_iter max_iter tol1 tol2 lambda_w prox_w lambda_h prox_h')
    Results = namedtuple('Results', 'w h i obj_history experiment')

    # experiment parameters
    experiment = Experiment(method='admm',
                            components=k,
                            rho=rho,
                            distance_type=distance_type,
                            nndsvd_init=nndsvd_init,
                            min_iter=min_iter,
                            max_iter=max_iter,
                            tol1=tol1,
                            tol2=tol2,
                            lambda_w=reg_w[0],
                            prox_w=reg_w[1],
                            lambda_h=reg_h[0],
                            prox_h=reg_h[1])


    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # initialize
    w, h, w_aux, h_aux, dual_w, dual_h, v_aux, dual_v = initialize(v, k, nndsvd_init)

    # initial distance value
    obj_history = [distance(v, w@h, distance_type=distance_type)]

    # Main iteration
    for i in range(max_iter):

        if distance_type == 'eu':
            h_aux = aux_update(h, dual_h, w_aux, v, None, rho, distance_type)
            w_aux = aux_update(w.T, dual_w.T, h_aux.T, v.T, None, rho, distance_type)
            w_aux = w_aux.T

            h = prox(reg_h[1], h_aux, dual_h, rho=rho, lambda_=reg_h[0])
            w = prox(reg_w[1], w_aux.T, dual_w.T, rho=rho, lambda_=reg_w[0])
            w = w.T

        elif distance_type == 'kl':
            h_aux = aux_update(h, dual_h, w_aux, v_aux, dual_v, rho, distance_type)
            w_aux = aux_update(w.T, dual_w.T, h_aux.T, v_aux.T, dual_v.T, rho, distance_type)
            w_aux = w_aux.T

            h = prox(reg_h[1], h_aux, dual_h, rho=rho, lambda_=reg_h[0])
            w = prox(reg_w[1], w_aux.T, dual_w.T, rho=rho, lambda_=reg_w[0])
            w = w.T

            v_bar = w_aux @ h_aux - dual_v
            v_aux = 1/2 * ((v_bar-1) + np.sqrt((v_bar-1)**2 + 4*v))

            dual_v = dual_v + v_aux - w_aux @ h_aux

        else:
            raise TypeError('Unknown loss type.')

        dual_h = dual_h + h - h_aux
        dual_w = dual_w + w - w_aux

        # Iteration info
        obj_history.append(distance(v, w@h, distance_type=distance_type))
        print('[{}]: {:.{}f}'.format(i, obj_history[-1], tol_precision))

        # Check convergence; save and break iteration
        if i > min_iter:
            # unpacking the last to entries of obj_history and reversing order
            converged = convergence_check(*obj_history[-2:][::-1], tol1, tol2)
            if converged:
                results = Results(w=w, h=h, i=i, obj_history=obj_history, experiment=experiment)
                logging.warning('Converged.')
                return results

        # save every XX iterations
        # if i % 100 == 0:
        #     save_results(save_str, w, h, i, obj_history, experiment_dict)

    else:
        # save on max_iter
        logging.info('Max iteration reached.')

    results = Results(w=w, h=h, i=i, obj_history=obj_history, experiment=experiment)
    return results