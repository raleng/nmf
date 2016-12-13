# File handling
load_file = 'data/pet-matlab/phantom2.mat'
load_var = 'sinodata'

save_file = 'nmf_exact'
save_dir = './results/phantom2'

# NMF Variables
features = 3
kl = True

# Iteration/Algorithm Variables
max_iter = 100000
tol1 = 1e-3
tol2 = 1e-3
alpha_w = 0.0
alpha_h = 0.0
