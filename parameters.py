# either mur, anls, admm, admm_nnls, or ao_admm
method = 'ao_admm_local'

# NMF Variables
features = [5]
lambda_w = [1]
lambda_h = [100]

# Phantom
phantom = 'phantom1'
phantom_version = 'exact'  # exact / noise

# Iteration/Algorithm Variables
min_iter = 25
max_iter = 10000
tol1 = 1e-3
tol2 = 1e-3

# method specifics
if method == 'mur':
    distance_type = 'eu'
    nndsvd_init = True

elif method == 'anls':
    use_fcnnls = False

elif method == 'admm_nnls':
    rho = [100]
    use_fcnnls = False

elif method == 'admm':
    distance_type = 'eu'
    rho = [100]

    prox_w = 'nn'
    prox_h = 'l2n'
    if prox_w == 'nn':
        lambda_w = [0]
    if prox_h == 'nn':
        lambda_h = [0]

elif method == 'ao_admm':
    distance_type = 'eu'
    loss_type = 'ls'
    admm_iter = 10

    prox_w = 'l1inf_transpose'
    prox_h = 'nn'
    if prox_w == 'nn':
        lambda_w = [0]
    if prox_h == 'nn':
        lambda_h = [0]

elif method == 'ao_admm_local':
    distance_type = 'eu'
    loss_type = 'ls'
    admm_iter = 10

    prox_w = 'nn'
    prox_h = 'l2n'
    if prox_w == 'nn':
        lambda_w = [0]
    if prox_h == 'nn':
        lambda_h = [0]

else:
    raise Exception('Unknown method: {}.'.format(method))

# File handling
load_dir = 'data/pet-matlab/'
load_file = load_dir + phantom + '.mat'

if phantom_version == 'noise':
    load_var = 'sinodata_noise'
elif phantom_version == 'exact':
    load_var = 'sinodata_exact'
else:
    raise Exception('Unknown dataset: {}.'.format(phantom_version))

save_dir = './results/{}_{}/{}'.format(phantom, phantom_version, method)
#save_dir = './results/admm_reg_test'
