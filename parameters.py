# NMF Variables
features = [3]
use_fcnnls = True
lambda_w = [0, 1, 10, 100]
lambda_h = [0, 1, 10, 100]

# either mur, anls, admm, or admm_nnls
method = 'admm_nnls'

# Phantom
phantom = 'phantom2'
phantom_version = 'exact'  # exact / noise

# Iteration/Algorithm Variables
min_iter = 100
max_iter = 10000
tol1 = 1e-5
tol2 = 1e-5

# File handling
load_dir = 'data/pet-matlab/'
load_file = load_dir + phantom + '.mat'

if phantom_version == 'noise':
    load_var = 'sinodata_noise'
elif phantom_version == 'exact':
    load_var = 'sinodata_exact'
else:
    raise Exception('Unknown dataset: {}.'.format(phantom_version))

save_name = 'nmf_{meth}_{feat}'.format(
    meth=method,
    feat=features,
    )
save_dir = './results/{}_{}/{}'.format(phantom, phantom_version, method)

# method specifics
if method == 'mur':
    distance_type = 'eu'
    save_file = '{name}_{dist}'.format(
        name=save_name,
        dist=distance_type,
    )

elif method == 'anls':
    save_file = save_name

elif method in {'admm', 'admm_nnls'}:
    rho = [1, 10, 10, 100]
    save_file = '{name}_{rho}'.format(
        name=save_name,
        rho=rho,
    )

elif method == 'admm_nnls':
    rho = [1]
    save_file = save_name

else:
    raise Exception('Unknown method: {}.'.format(method))
