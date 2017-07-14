import os
from importlib import import_module

import begin
# noinspection PyUnresolveReferences
import better_exceptions
import numpy as np

from misc import loadme


def cssls(cTc, cTa, *, p_set=None):
    """ """
    k = np.zeros((cTa.shape))
    if p_set is None or np.all(p_set):
        k = np.linalg.solve(cTc, cTa)
    else:
        l_var, p_rhs = p_set.shape

        # sort passive set
        coded_p_set = [2**x for x in range(l_var-1, -1, -1)] @ p_set
        sorted_e_set = np.argsort(coded_p_set)
        sorted_p_set = coded_p_set[sorted_e_set]

        # since the passive set is sorted, if breaks contains a 0, that means
        # the passive sets were equal
        breaks = np.diff(sorted_p_set)
        zeros, = np.nonzero(breaks)
        break_idx = [0, *list(zeros), p_rhs]

        for k in range(len(break_idx)):
            cols2solve = sorted_e_set[break_idx[k]:break_idx[k]+1]
            varis = p_set[:, sorted_e_set[break_idx[k]]]
            k[varis, cols2solve] = np.linalg.solve(cTc[np.ix_(varis, varis)],
                                                   cTa[np.ix_(varis, cols2solve)])

    return k


def fcnnls(c, a):
    """ """

    # init
    n_obs, l_var = c.shape
    p_rhs = a.shape[1]
    w = np.zeros(l_var, p_rhs)
    iter = 0
    max_iter = 3 * l_var

    # precompute parts of pseudoinverse
    cTc = c.T @ c
    cTa = c.T @ a
     
    # obtain the initial feasible solution and corresponding passive set
    k = cssls(cTc, cTa)
    p_set = k > 0

    k[~p_set] = 0
    d = k
    f_set = np.nonzero(~np.all(p_set))

    # active set algorithm for nnls main loop
    while f_set == []:

        # solve for the passive variables (uses subroutine cssls)
        k[:, f_set] = cssls(cTc, cTa[:, f_set], p_set=p_set[:, f_set])
        
        # find any infeasible solutions
        h_set = f_set[]
        
        # make infeasible solutions feasible (standard nnls inner loop)
        if not h_set == []:
            nh_set = len(h_set)
            alpha = np.zeros((l_var, nh_set))

            while not h_set == [] and counter < max_iter:
                counter += 1
                alpha[:, 1:nh_set] = np.inf
                
                # find indices of negative variables in passive set
                i, j =

                h_idx = np.ravel_multi_index()
                neg_idx = np.ravel_multi_index()
                alpha[h_idx] = d[neg_idx] / ( d[neg_idx] - k[neg_idx] )

                min_idx = np.argmin(alpha[:, 1:nh_set])
                alpha_min = alpha[min_idx]

                alpha[:, 1:nh_set] = repmat
                d[:, h_set] = d[:, h_set] - alpha[:, 1:nh_set] * (d[:, h_set] - k[:, h_set])

                idx_to_zero = np.ravel_multi_index()
                d[idx_to_zero] = 0

                p_set[idx_to_zero] = 0
                k[:, h_set] = cssls(cTc, cTa[:, h_set], p_set=p_set[:, h_set])

        # make sure the solution has converged
        if counter == max_iter:
            print('too bad, max iter')

        # check solutions for optimality
        w[:, f_set] = cTa[:, f_set] - cTc @ k[:, f_set]
        j_set = find
        f_set = np.setdiff1d(f_set, f_set[j_set])

        # for non-optimal solutions, add the appropriate variable to Pset
        if not f_set == []:
            foo = ~p_set[:, f_set]*w[:, f_set]
            mx_idx = np.argmax(foo)
            mx = foo[mx_idx]
            p_set[np.ravel_multi_index()] = 1
            d[:, f_set] = k[:, f_set]


@begin.start
def main(param_file='parameters_admm_reg'):
    """ NMF with FCNNLS """

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

    fcnnls(data,
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
