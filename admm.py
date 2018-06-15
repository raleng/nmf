# system imports
import os
# noinspection PyUnresolvedReferences
import better_exceptions

# math imports
import numpy as np

# personal imports
from utils import convergence_check, distance, nndsvd, save_results


def initialize(data, features):
    """ Initializing variables """

    w, h = nndsvd(data, features)
    x = w @ h

    w_p = w.copy()
    h_p = h.copy()

    alpha_x = np.zeros_like(x)
    alpha_w = np.zeros_like(w)
    alpha_h = np.zeros_like(h)

    return x, w, h, w_p, h_p, alpha_x, alpha_w, alpha_h


def w_update(x, h, w_p, alpha_x, alpha_w, rho):
    """ ADMM update of W """
    a = h @ h.T + np.eye(h.shape[0])
    b = h @ x.T + w_p.T + 1/rho * (h @ alpha_x.T - alpha_w.T)
    w = np.linalg.solve(a, b).T
    return w


def h_update(x, w, h_p, alpha_x, alpha_h, rho):
    """ ADMM update of H """
    a = w.T @ w + np.eye(w.shape[1])
    b = w.T @ x + h_p + 1/rho * (w.T @ alpha_x - alpha_h)
    h = np.linalg.solve(a, b)
    return h


def x_update(v, wh, alpha_x, rho, distance_type='kl'):
    """ ADMM update of X """
    if distance_type == 'kl':
        value = rho * wh - alpha_x - 1
        x = value + np.sqrt(value**2 + 4*rho*v)
        x /= 2*rho
    elif distance_type == 'eu':
        x = wh + (v - wh)
    else:
        raise KeyError('Unknown distance type.')

    return x


def wh_p_update(w, h, alpha_w, alpha_h, rho):
    """ ADMM update of W_plus and H_plus """
    w_p = np.maximum(w + 1/rho * alpha_w, 0)
    h_p = np.maximum(h + 1/rho * alpha_h, 0)
    return w_p, h_p


def alpha_update(x, w, h, wh, w_p, h_p, alpha_x, alpha_w, alpha_h, rho):
    """ ADMM update dual variables """
    alpha_x = alpha_x + rho * (x - wh)
    alpha_h = alpha_h + rho * (h - h_p)
    alpha_w = alpha_w + rho * (w - w_p)
    return alpha_x, alpha_h, alpha_w


def admm(v, k, *, rho=1, distance_type='kl', min_iter=10, max_iter=100000, tol1=1e-3,
         tol2=1e-3, save_dir='./results/'):
    """ NMF with ADMM

    Expects following arguments:
    v -- 2D data
    k -- number of components

    Accepts keyword arguments:
    rho -- FLOAT:
    min_iter -- INT: minimum number of iterations (default: 10)
    max_iter -- INT: maximum number of iterations (default: 100000)
    tol1 -- FLOAT:
    tol2 -- FLOAT:
    save_dir -- STRING: folder to which to save
    save_file -- STRING: file name to which to save
    """

    # create folder, if not existing
    os.makedirs(save_dir, exist_ok=True)
    save_name = 'nmf_admm_{feat}_{rho}_{dist}'.format(
        feat=k,
        rho=rho,
        dist=distance_type,
    )
    save_str = os.path.join(save_dir, save_name)

    # save all parameters in dict; to be saved with the results
    experiment_dict = {'k': k,
                       'max_iter': max_iter,
                       'rho': rho,
                       }

    # used for cmd line output; only show reasonable amount of decimal places
    tol = min(tol1, tol2)
    tol_precision = int(format(tol, 'e').split('-')[1]) if tol < 1 else 2

    x, w, h, w_p, h_p, alpha_x, alpha_w, alpha_h = initialize(v, k)

    # initial distance value
    obj_history = [distance(v, w@h)]

    # Main iteration
    for i in range(max_iter):

        # Update step
        w = w_update(x, h, w_p, alpha_x, alpha_w, rho)
        h = h_update(x, w, h_p, alpha_x, alpha_h, rho)
        wh = w @ h

        x = x_update(v, wh, alpha_x, rho, distance_type)
        w_p, h_p = wh_p_update(w, h, alpha_w, alpha_h, rho)
        alpha_x, alpha_h, alpha_w, = alpha_update(x, w, h, wh, w_p, h_p, alpha_x, alpha_w,
                                                  alpha_h, rho)

        # Iteration info
        obj_history.append(distance(v, w_p@h_p))
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
