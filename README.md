# Non-negative Matrix Factorization

This package implements four ways to compute a non-negative matrix factorization of a 2D non-negative numpy array.

1. **Multiplicative update rules** (MUR)
2. **Alternating non-negative least squares** (ANLS)
3. **Alternating direction method of multipliers** (ADMM)
4. **Alternating optimization ADMM** (AO-ADMM)

## Usage

Given non-negative `data` and the number of `components` you want the dataset to factorize into, you simply create an NMF instance and use the factorize method to compute the factorization.

```
$ from nmf import NMF
$ nmf = NMF(data, components)
$ nmf.factorize(method='mur', **method_args)
```

You can directly access the factors `nmf.w` and `nmf.h`.

## Saving

If you want to save the results to a file, you can use the `save_factorization` method.
The default folder to save is `./results` and the default file name is constructed using the parameters used in the factorization.
