import numpy as np


def initialize(c, a):
    """ variables init """
    n_obs, l_var = c.shape
    p_rhs = a.shape[1]
    w = np.zeros((l_var, p_rhs))
    iter_counter = 0
    iter_max = 3 * l_var
    return n_obs, l_var, p_rhs, w, iter_counter, iter_max


def cssls(cTc, cTa, *, p_set=None):
    """ """

    k = np.zeros(cTa.shape)
    if p_set is None or np.all(p_set):
        k = np.linalg.solve(cTc, cTa)
    else:
        l_var, p_rhs = p_set.shape

        # sort passive set
        coded_p_set = np.array([2**x for x in range(l_var-1, -1, -1)]) @ p_set
        sorted_e_set = np.argsort(coded_p_set)
        sorted_p_set = coded_p_set[sorted_e_set]

        # since the passive set is sorted, if breaks contains a 0, that means
        # the passive sets were equal
        breaks = np.diff(sorted_p_set)
        zeros, = np.nonzero(breaks)
        break_idx = [0, *list(zeros), p_rhs]

        for item in range(len(break_idx)-1):
            print('cssls loop: {}'.format(item))
            cols2solve = sorted_e_set[break_idx[item]+1:break_idx[item+1]]
            # varis = p_set[:, sorted_e_set[break_idx[item]+1]]
            # k[varis, cols2solve] = np.linalg.solve(cTc, cTa[varis, cols2solve])
            k[:, cols2solve] = np.linalg.solve(cTc, cTa[:, cols2solve])

    return k


def fcnnls(c, a):
    """  solves: min_K>0 || CK - A || """

    n_obs, l_var, p_rhs, w, iter_counter, iter_max = initialize(c, a)

    # precompute parts of pseudoinverse - step 3
    cTc = c.T @ c
    cTa = c.T @ a

    # obtain the initial feasible solution and corresponding passive set
    print('First cssls')
    k = cssls(cTc, cTa)  # step 4
    p_set = k > 0  # step 5
    k[~p_set] = 0  # step 7
    d = k.copy()
    f_set, = np.nonzero(~np.all(p_set, 0))  # step 6

    # active set algorithm for nnls main loop
    while f_set.size != 0:

        # solve for the passive variables (uses subroutine cssls)
        print('Second cssls')
        k[:, f_set] = cssls(cTc, cTa[:, f_set], p_set=p_set[:, f_set])
        
        # find any infeasible solutions
        f_nz, = np.nonzero(np.any(k[:, f_set] < 0, 0))
        h_set = f_set[f_nz]  # step 10
        
        # make infeasible solutions feasible (standard nnls inner loop)
        if h_set.size != 0:
            nh_set = h_set.size
            alpha = np.zeros((l_var, nh_set))

            while h_set.size != 0 and iter_counter < iter_max:
                iter_counter += 1
                alpha[:, range(nh_set)] = np.inf
                
                # find indices of negative variables in passive set
                i, j = np.nonzero(np.logical_and(p_set[:, h_set], k[:, h_set] < 0))
                h_idx = np.ravel_multi_index((i, j), (l_var, nh_set))
                neg_idx = np.ravel_multi_index((i, h_set[j]), k.shape)

                alpha.flat[h_idx] = d.flat[neg_idx] / (d.flat[neg_idx] - k.flat[neg_idx])
                min_idx = np.argmin(alpha[:, range(nh_set)])
                alpha_min = alpha.flat[min_idx]

                alpha[:, range(nh_set)] = np.tile(alpha_min, [l_var, 1])
                d[:, h_set] -= alpha[:, range(nh_set)] * (d[:, h_set] - k[:, h_set])

                idx_to_zero = np.ravel_multi_index((min_idx, h_set), d.shape)
                d.flat[idx_to_zero] = 0
                p_set.flat[idx_to_zero] = 0

                print('third cssls')
                k[:, h_set] = cssls(cTc, cTa[:, h_set], p_set=p_set[:, h_set])
                h_set, = np.nonzero(np.any(k < 0, 0))
                nh_set = h_set.size

        # make sure the solution has converged
        if iter_counter == iter_max:
            print('too bad, max iter')

        # check solutions for optimality
        w[:, f_set] = cTa[:, f_set] - cTc @ k[:, f_set]
        optimal = ~p_set[:, f_set] * w[:, f_set] <= 0
        j_set, = np.nonzero(np.all(optimal, 0))
        f_set = np.setdiff1d(f_set, f_set[j_set])

        # for non-optimal solutions, add the appropriate variable to Pset
        if f_set.size != 0:
            non_optimal = ~p_set[:, f_set] * w[:, f_set]
            mx_idx = np.argmax(non_optimal)
            p_set.flat[np.ravel_multi_index((mx_idx, f_set), (l_var, p_rhs))] = 1
            d[:, f_set] = k[:, f_set]

    return k
