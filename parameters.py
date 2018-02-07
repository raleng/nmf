# NMF Variables
features = 2
use_fcnnls = True
lambda_w = 1
lambda_h = 100

# either admm or anls
method = 'admm'

# Phantom
phantom = 'phantom2'
phantom_version = 'noise' # exact / noise

# Iteration/Algorithm Variables
min_iter = 500
max_iter = 1200
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

save_name = 'nmf_{meth}_{feat}_{lambda_w}_{lambda_h}'.format(
    meth=method,
    feat=features,
    lambda_w=lambda_w,
    lambda_h=lambda_h,
    )
save_dir = './results/{}_{}/{}'.format(phantom, phantom_version, method)

# method specifics
if method == 'anls':
    save_file = save_name
elif method == 'admm':
    rho = 10
    save_file = '{name}_{rho}'.format(
        name=save_name,
        rho=rho,
        )
else:
    raise Exception('Unknown method: {}.'.format(method))
