import numpy as np


def distance(x, wh):
    """ Kullback-Leibler divergence """
    # value = 0.5 * np.sum((x - wh) ** 2)
    value = x * np.log(x / wh)
    value = np.where(np.isnan(value), 0, value)
    value = np.sum(value - x + wh)
    return value


def nndsvd(x, rank=None):
    """ svd based nmf initialization """

    u, s, v = np.linalg.svd(x, full_matrices=False)
    v = v.T

    if rank is None:
        rank = x.shape[1]

    w = np.zeros((x.shape[0], rank))
    h = np.zeros((rank, x.shape[1]))

    w[:, 0] = np.sqrt(s[0]) * np.abs(u[:, 0])
    h[0, :] = np.sqrt(s[0]) * np.abs(v[:, 0].T)

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

    return w, h


def save_results(save_str, w, h, i, obj_history, experiment_dict):
    """ save results """

    # Normalizing
    # h, norm = normalize(h, return_norm=True)
    # w = w * norm

    np.savez(save_str, w=w, h=h, i=i, obj_history=obj_history,
             experiment_dict=experiment_dict)
    print('Results saved in {}.'.format(save_str))
