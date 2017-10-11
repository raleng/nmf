# system imports
import begin
import os
from importlib import import_module
# noinspection PyUnresolvedReferences
# import better_exceptions

# math imports
from math import sqrt
import numpy as np
from scipy import optimize

# personal imports
import fcnnls
from misc import loadme
from utils import distance, nndsvd, save_results


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


def anls(x, k, *, use_fcnnls=False, lambda_w=0, lambda_h=0, max_iter=1000, tol1=1e-3,
         tol2=1e-3, save_dir='./results/', save_file='nmf_anls'):
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
    save_str = os.path.join(save_dir, save_file)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {
        'k': k,
        'max_iter': max_iter,
        'lambda_w': lambda_w,
        'lambda_h': lambda_h,
        'tol1': tol1,
        'tol2': tol2,
    }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    # Random W, H init
    # w = np.random.rand(x.shape[0], k)
    # h = np.random.rand(k, x.shape[1])

    # NNDSVD W, H init
    w, h = nndsvd(x, k)

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

        if i > 10:
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

    if params.use_fcnnls:
        print('Using FCNNLS.')

    anls(
        data,
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
