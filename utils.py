import numpy as np


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


def distance(x, wh, distance_type='eu'):
    """ distance function for Kullback-Leibler divergence and Euclidean distance """

    if distance_type == 'kl':
        """ Kullback-Leibler divergence """
        value = x * np.log(x / wh)
        value = np.where(value == np.inf, 0, value)
        value = np.where(np.isnan(value), 0, value)
        value = np.sum(value - x + wh)
    elif distance_type == 'eu':
        """ Euclidean distance """
        value = 0.5 * np.sum((x - wh) ** 2)
    else:
        raise KeyError('Distance type unknown: use "kl" or "eu"')

    return value


def nndsvd(x, rank=None, variant='zero'):
    """ svd based nmf initialization

    Paper:
        Boutsidis, Gallopoulos: SVD based initialization: A head start for
        nonnegative matrix factorization
    """

    u, s, v = np.linalg.svd(x, full_matrices=False)
    v = v.T

    if rank is None:
        rank = x.shape[1]

    # initialize w, h
    w = np.zeros((x.shape[0], rank))
    h = np.zeros((rank, x.shape[1]))

    # first column/row: dominant singular triplets of x
    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

    # find dominant singular triplets for every unit rank matrix u_i * v_i^T
    # see Paper, page 8, for details
    for i in range(1, rank):
        ui = u[:, i]
        vi = v[:, i]

        ui_pos = (ui >= 0) * ui
        ui_neg = (ui < 0) * -ui
        vi_pos = (vi >= 0) * vi
        vi_neg = (vi < 0) * -vi

        ui_pos_norm = np.linalg.norm(ui_pos, 2)
        ui_neg_norm = np.linalg.norm(ui_neg, 2)
        vi_pos_norm = np.linalg.norm(vi_pos, 2)
        vi_neg_norm = np.linalg.norm(vi_neg, 2)

        norm_pos = ui_pos_norm * vi_pos_norm
        norm_neg = ui_neg_norm * vi_neg_norm

        if norm_pos >= norm_neg:
            w[:, i] = np.sqrt(s[i] * norm_pos) / ui_pos_norm * ui_pos
            h[i, :] = np.sqrt(s[i] * norm_pos) / vi_pos_norm * vi_pos.T
        else:
            w[:, i] = np.sqrt(s[i] * norm_neg) / ui_neg_norm * ui_neg
            h[i, :] = np.sqrt(s[i] * norm_neg) / vi_neg_norm * vi_neg.T

    if variant == 'mean':
        w = np.where(w == 0, np.mean(x), w)
        h = np.where(h == 0, np.mean(x), h)
    elif variant == 'random':
        random_matrix = np.mean(x) * np.random.random_sample(w.shape) / 100
        w = np.where(w == 0, random_matrix, w)
        random_matrix = np.mean(x) * np.random.random_sample(h.shape) / 100
        h = np.where(h == 0, random_matrix, h)

    return w, h


def save_results(save_str, w, h, i, obj_history, experiment_dict):
    """ save results """

    # Normalizing
    # h, norm = normalize(h, return_norm=True)
    # w = w * norm

    np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
             experiment_dict=experiment_dict)
    print('Results saved in {}.'.format(save_str))
