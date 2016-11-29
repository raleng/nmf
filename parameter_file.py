# File handling
load_file = 'data/pet-matlab/no_noise_data.mat'
load_var = 'uexact'

save_file = 'nmf_exact'
save_dir = './results/'

# NMF Variables
features = 3
kl = False

# Iteration/Algorithm Variables
max_iter = 100000
tol1 = 1e-3
tol2 = 1e-3
alpha_w = 0.0
alpha_h = 0.0
