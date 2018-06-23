import numpy as np


def initialize(c, a):
    """ variables init """
    n_obs, l_var = c.shape
    p_rhs = a.shape[1]
    w = np.zeros((l_var, p_rhs))
    iter_counter = 0
    iter_max = 3 * l_var
    return n_obs, l_var, p_rhs, w, iter_counter, iter_max


def cssls(ct_c, ct_a, *, p_set=None):
    """ combinatorial subspace least squares

    solve the set of equations cTa = cTc * k for the variables in set p_set
    using fast combinatorial approach
    """

    k = np.zeros_like(ct_a)
    if p_set is None or np.all(p_set):
        k = np.linalg.solve(ct_c, ct_a)
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
        break_idx = [-1] + list(zeros) + [p_rhs-1]

        for index1, index2 in zip(break_idx[:-1], break_idx[1:]):
            # this test is necessary because [i:i] is an empty list in python
            if index1+1 == index2:
                cols2solve = [sorted_e_set[index2]]
            else:
                # index1+1 because np.diff finds breaks infront of the break
                # index2+1 index2 needs to be included
                cols2solve = sorted_e_set[index1+1:index2+1]

            varis = p_set[:, sorted_e_set[index1+1]]
            grid_vc = np.ix_(varis, cols2solve)
            grid_vv = np.ix_(varis, varis)
            k[grid_vc] = np.linalg.solve(ct_c[grid_vv], ct_a[grid_vc])

    return k


def fcnnls(c, a):
    """  solves: min_K>0 || CK - A || 
    
    Paper:
        Van Benthem, Keenan: Fast algorithm for the solution of large-scale
        non-negativity constrained least squares problems

    See Paper, page 446, for "step" references

    This is basically a conversion of the MatLab Code at the end of the above paper
    """

    n_obs, l_var, p_rhs, w, iter_counter, iter_max = initialize(c, a)

    # precompute parts of pseudoinverse - step 3
    ct_c = c.T @ c
    ct_a = c.T @ a

    # obtain the initial feasible solution and corresponding passive set
    k = cssls(ct_c, ct_a)  # step 4
    p_set = k > 0  # step 5
    k[~p_set] = 0  # step 7
    d = k.copy()
    f_set, = np.nonzero(~np.all(p_set, 0))  # step 6

    # active set algorithm for nnls main loop
    while f_set.size != 0:
        # solve for the passive variables (uses subroutine cssls)
        k[:, f_set] = cssls(ct_c, ct_a[:, f_set], p_set=p_set[:, f_set])

        # find any infeasible solutions
        f_nz, = np.nonzero(np.any(k[:, f_set] < 0, 0))
        h_set = f_set[f_nz]  # step 10

        # make infeasible solutions feasible (standard nnls inner loop)
        if h_set.size != 0:
            nh_set = h_set.size
            alpha = np.zeros((l_var, nh_set))

            # see paper for details, inner loop VERY technical
            while h_set.size != 0 and iter_counter < iter_max:
                iter_counter += 1
                alpha[:, range(nh_set)] = np.inf

                # find indices of negative variables in passive set
                i, j = np.nonzero(np.logical_and(p_set[:, h_set], k[:, h_set] < 0))
                h_idx = np.ravel_multi_index((i, j), (l_var, nh_set))
                neg_idx = np.ravel_multi_index((i, h_set[j]), k.shape)

                alpha.flat[h_idx] = d.flat[neg_idx] / (d.flat[neg_idx] - k.flat[neg_idx])
                min_idx = np.argmin(alpha[:, range(nh_set)], axis=0)
                alpha_min = alpha.flat[min_idx]

                alpha[:, range(nh_set)] = np.tile(alpha_min, [l_var, 1])
                d[:, h_set] -= alpha[:, range(nh_set)] * (d[:, h_set] - k[:, h_set])

                idx_to_zero = np.ravel_multi_index((min_idx, h_set), d.shape)
                d.flat[idx_to_zero] = 0
                p_set.flat[idx_to_zero] = 0  # step 12

                k[:, h_set] = cssls(ct_c, ct_a[:, h_set], p_set=p_set[:, h_set])  # stp 13
                h_set, = np.nonzero(np.any(k < 0, 0))
                nh_set = h_set.size

        # make sure the solution has converged
        if iter_counter == iter_max:
            print('Not converged.')

        # check solutions for optimality
        w[:, f_set] = ct_a[:, f_set] - ct_c @ k[:, f_set]
        optimal = ~p_set[:, f_set] * w[:, f_set] <= 0
        j_set, = np.nonzero(np.all(optimal, 0))
        f_set = np.setdiff1d(f_set, f_set[j_set])

        # for non-optimal solutions, add the appropriate variable to Pset
        if f_set.size != 0:
            non_optimal = ~p_set[:, f_set] * w[:, f_set]
            mx_idx = np.argmax(non_optimal, axis=0)
            p_set.flat[np.ravel_multi_index((mx_idx, f_set), (l_var, p_rhs))] = 1
            d[:, f_set] = k[:, f_set]

    return k
