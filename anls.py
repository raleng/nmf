# system imports
import os
# noinspection PyUnresolvedReferences
# import better_exceptions

# math imports
from math import sqrt
import numpy as np
from scipy import optimize

# personal imports
import fcnnls
from utils import convergence_check, distance, nndsvd, save_results


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
            h[:, i], _ = optimize.nnls(a, b[:, i])

    return h


def anls(x, k, *, use_fcnnls=False, lambda_w=0, lambda_h=0, min_iter=10, max_iter=1000,
         tol1=1e-3, tol2=1e-3, nndsvd_init=(True, 'zero'), save_dir='./results/'):
    """ NMF via ANLS with FCNNLS

    according to the follow papers:
    - Kim, Park: Non-negative matrix factorization based on alternating non-negativity
        constrained least squares and active set method

    fcnnls paper:
    - Benthem, Keenan: Fast algorithm for the solution of large-scale non-negativity-
        constrained least squares problems

    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_anls_{feat}_{lambda_w}_{lambda_h}'.format(
        feat=k,
        lambda_w=lambda_w,
        lambda_h=lambda_h,
    )
    if nndsvd_init[0]:
        save_name += '_nndsvd{}'.format(nndsvd_init[1][0])
    else:
        save_name += '_random'

    if use_fcnnls:
        save_name = '{}_fcnnls'.format(save_name)

    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {
        'k': k,
        'max_iter': max_iter,
        'lambda_w': lambda_w,
        'lambda_h': lambda_h,
        'tol1': tol1,
        'tol2': tol2,
        'fcnnls': use_fcnnls,
    }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    if nndsvd_init[0]:
        w, h = nndsvd(x, k, variant=nndsvd_init[1])
    else:
        w = np.random.rand(x.shape[0], k)
        h = np.random.rand(k, x.shape[1])

    # sc_init = stop_criterium(x, w, h, lambda_w, lambda_h)

    obj_history = [1e10]
    # MAIN ITERATION
    for i in range(max_iter):

        # Update step
        w = w_update(x, h, lambda_w, use_fcnnls=use_fcnnls)
        h = h_update(x, w, lambda_h, use_fcnnls=use_fcnnls)

        # Iteration info
        obj_history.append(distance(x, w@h))
        print('[{}]: {:.{}f}'.format(i, obj_history[-1], tol_precision))

        if i > min_iter:
            converged = convergence_check(obj_history[-1], obj_history[-2], tol1, tol2)
            if converged:
                save_results(save_str, w, h, i, obj_history, experiment_dict)
                print('Converged.')
                break

        if i % 100 == 0:
            save_results(save_str, w, h, i, obj_history, experiment_dict)

    else:
        save_results(save_str, w, h, max_iter, obj_history, experiment_dict)
        print('Max iteration reached.')
