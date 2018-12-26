# Non-negative Matrix Factorization

This package implements four ways to compute a non-negative matrix factorization of a 2D non-negative numpy array.

1. **Multiplicative update rules** (MUR)
2. **Alternating non-negative least squares** (ANLS)
3. **Alternating direction method of multipliers** (ADMM)
4. **Alternating optimization ADMM** (AO-ADMM)

## Usage

### Compute factorization

Given non-negative `data` and the number of `components` you want the dataset to factorize into, you simply create an NMF instance and use the factorize method to compute the factorization.

```
$ from nmf import NMF
$ nmf = NMF(data, components)
$ nmf.factorize(method='mur', **method_args)
```

You can directly access the factors `nmf.w` and `nmf.h`.

### Saving

If you want to save the results to a file, you can use the `save_factorization` method.
The default folder to save is `./results` and the default file name is constructed using the parameters used in the factorization.

## Methods

#### MUR

Following the papers:
* Lee, Seung: Learning the parts of objects by non-negative matrix factorization, 1999
* Lee, Seung: Algorithms for non-negative matrix factorization, 2001 

Accepts following method parameters:
* `distance_type` -- STRING: 'eu' for Euclidean, 'kl' for Kullback Leibler
* `min_iter` -- INT: minimum number of iterations
* `max_iter` -- INT: maximum number of iterations
* `tol1` -- FLOAT: convergence tolerance
* `tol2` -- FLOAT: convergence tolerance
* `lambda_w` -- FLOAT: regularization parameter for w-Update
* `lambda_h` -- FLOAT: regularization parameter for h-Update
* `nndsvd_init` -- Tuple(BOOL, STRING): if BOOL = True, use NNDSVD-type STRING
* `save_dir` -- STRING: folder to which to save

#### ANLS

Following the papers:
* Kim, Park: Non-negative matrix factorization based on alternating non-negativity constrained least squares and active set method

Accepts following method parameters:
* `distance_type` -- STRING: 'eu' for Euclidean, 'kl' for Kullback Leibler
* `use_fcnnls` -- BOOL: if true, use FCNNLS algorithm
* `lambda_w` -- FLOAT: regularization parameter for w-Update
* `lambda_h` -- FLOAT: regularization parameter for h-Update
* `min_iter` -- INT: minimum number of iterations
* `max_iter` -- INT: maximum number of iterations
* `tol1` -- FLOAT: convergence tolerance
* `tol2` -- FLOAT: convergence tolerance
* `nndsvd_init` -- Tuple(BOOL, STRING): if BOOL = True, use NNDSVD-type STRING
* `save_dir` -- STRING: folder to which to save

#### ADMM

Following the papers:
* Huang, Sidiropoulos, Liavas: A flexible and efficient algorithmic framework for constrained matrix and tensor factorization, 2015

Accepts following method parameters:
* `rho` -- FLOAT: ADMM dampening parameter
* `distance_type` -- STRING: 'eu' for Euclidean, 'kl' for Kullback Leibler
* `reg_w` -- Tuple(FLOAT, STRING): value und type of w-regularization
* `reg_h` -- Tuple(FLOAT, STRING): value und type of h-regularization
* `min_iter` -- INT: minimum number of iterations
* `max_iter` -- INT: maximum number of iterations
* `tol1` -- FLOAT: convergence tolerance
* `tol2` -- FLOAT: convergence tolerance
* `nndsvd_init` -- Tuple(BOOL, STRING): if BOOL = True, use NNDSVD-type STRING
* `save_dir` -- STRING: folder to which to save

#### AO-ADMM

Following the papers:
* Huang, Sidiropoulos, Liavas: A flexible and efficient algorithmic framework for constrained matrix and tensor factorization, 2015

Accepts following method parameters:
* `distance_type` -- STRING: 'eu' for Euclidean, 'kl' for Kullback Leibler
* `reg_w` -- Tuple(FLOAT, STRING): value und type of w-regularization
* `reg_h` -- Tuple(FLOAT, STRING): value und type of h-regularization
* `min_iter` -- INT: minimum number of iterations
* `max_iter` -- INT: maximum number of iterations
* `admm_iter` -- INT: maximum number of internal ADMM iterations
* `tol1` -- FLOAT: convergence tolerance
* `tol2` -- FLOAT: convergence tolerance
* `nndsvd_init` -- Tuple(BOOL, STRING): if BOOL = True, use NNDSVD-type STRING
* `save_dir` -- STRING: folder to which to save