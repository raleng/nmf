import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import solve_triangular


def initialize(cTb, dim_c, dim_b):
    """ Initialize variables """
    
    f = [set([]) for i in range(dim_b[1])]
    g = [set(range(dim_c[1])) for i in range(dim_b[1])]

    x = np.zeros((dim_c[1], dim_b[1]))
    y = -cTb

    alpha = [3 for i in range(dim_b[1])
    beta = [dim_c[1] + 1 for i in range(dim_b[1])

    return f, g, x, y, alpha, beta


def check_feasiblity(x, y):
    """ """

    I = set()
    V = []
    for j in range(x.shape[1]):
        if np.any(x[:, j] < 0) or np.any(y[:, j] < 0):
            I.add(j)
            x_index_set = set([i for i, val in enumerate(x[:, j] < 0) if val])
            y_index_set = set([i for i, val in enumerate(y[:, j] < 0) if val])
            V[j] = x_index_set.union(y_index_set)

    return I, V


def update_f_g(f, g, I, V):
    """ """

    V_hat = []
    for j in I:
        if len(V[j]) < beta[j]:
            beta[j] = np.sum(V[j])
            alpha[j] = 3
            V_hat[j] = V[j]
        elif len(V[j]) >= beta[j] and alpha[j] >= 1:
            alpha[j] -= 1
            V_hat[j] = V[j]
        elif len(V[j]) >= beta[j] and alpha[j] == 0:
            V_hat[j] = max(V[j])
        else:
            raise Exception

        f[j] = (f[j] - V_hat[j]).union(V_hat[j].intersection(g[j]))
        g[j] = (g[j] - V_hat[j]).union(V_hat[j].intersection(f[j]))

    return f, g


def column_grouping(I, f, g, cTc, cTb):
    """ """

    not_picked = I.copy()
    picked = set([])
    while len(not_picked) > 0: # iterate as long as indices are left
        current_min_index = min(not_picked) # start with the smallest index that is left
        template = f[current_min_index] # take the corresponding index set F for that smallest index
        # check all remaining indices, whether F(smallest) == F(index)
        # all those who are equal belong to one block that can be solved with the same cholesky decomp
        for index in not_picked:
            if template == f[index]:
                picked.add(index)

        # take only the relevant indices, that is all indices in F for the current template
        c_ixgrid = np.ix_(list(template), list(template))
        # solve_for_x returns only the F-version of x, save them to the "big" version according to index
        x_ixgrid = np.ix_(list(template), list(picked))
        x[x_ixgrid] = solve_for_x(cTc[c_ixgrid], cTb[list(template)])
        
        # for every index that has been picked at any one point, we have to compute y as well
        for index in picked:
            c_ixgrid = np.ix_(list(g[index]), list(f[index]))
            y_ixgrid = np.ix_(list(g[index]), [index])
            # SOLVE_FOR_Y returns only G-version of y, save them to the "big" version according to index
            y[y_ixgrid] = solve_for_y(cTc[c_ixgrid], cTb[list(g[index])], x[:, index])

        not_picked -= picked
        picked = set([])

    return x, y


def solve_for_x(A, B):
    """ solve normal equation via Cholesky decomposition and
    forward/backward substitution """
    L = cholesky(A)
    x = np.zeros(B.shape)
    for i in range(B.shape[1]):
        y = solve_triangular(L, B[:, i], lower=True)
        x[:, i] = solve_triangular(L, y, lower=True, trans=1)
    return x


def solve_for_y(A, B, x):
    """ """
    return A @ x - B


def check_convergence():
    """ """
    if True:
        return True
    else:
        return False


def bpp(c, b):
    """ Block principal pivoting algorithm

    solves the NNLS problem:

        min_(x>0) || CX - B ||_F^2

    """

    # precompute matrices
    cTc = c.T @ c
    cTb = c.T @ b

    # initialize variables
    f, g, x, y, alpha, beta = initialize(cTb, c.shape, b.shape)

    # iterate until x, y are feasable
    infeasible = True
    convergence = False
    while infeasible and not convergence:
        
        infeasible_indices, infeasible_var = check_feasibility(x, y) 

        f, g = update_f_g(f, g, infeasible_indices, infeasible_var)

        # calc x by column grouping
        x, y = column_grouping(infeasible_indices, f, g, cTc, cTb)

        convergence = check_convergence()

    return x
