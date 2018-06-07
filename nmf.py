#!/usr/bin/env python
import begin
from importlib import import_module
from itertools import product

import numpy as np

from misc import loadme


@begin.start
def main(param_file='parameters'):
    """ NMF with ANLS """

    # Parameter import
    try:
        params = import_module(param_file)
    except ImportError:
        print('No parameter file found.')
        return

    if params.phantom_version == 'noise':
        load_var = 'sinodata_noise'
    elif params.phantom_version == 'exact':
        load_var = 'sinodata_exact'
    else:
        raise Exception('Unknown dataset: {}.'.format(params.phantom_version))

    # Loading data
    try:
        if load_var == 'LOAD_MSOT':
            data = loadme.msot(params.load_file)
            print('Loaded MSOT data.')
        else:
            data = loadme.mat(params.load_file, params.load_var)
            print('Loaded PET data.')
    except AttributeError:
        print('No file/variable given.')
        return

    # Transform data dimensions
    if data.ndim == 3:
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]), order='F')
        print('Data was 3D. Reshaped to 2D.')

    # Logging info for FCNNLS usage
    if params.use_fcnnls and params.method in {'anls, admm_nnls'}:
        print('Using FCNNLS.')

    # Method call
    if params.method == 'mur':
        import mur
        for features, lambda_w, lambda_h in product(params.features,
                                                    params.lambda_w,
                                                    params.lambda_h):
            mur.mur(
                data,
                features,
                distance_type=params.distance_type,
                min_iter=params.min_iter,
                max_iter=params.max_iter,
                tol1=params.tol1,
                tol2=params.tol2,
                lambda_w=lambda_w,
                lambda_h=lambda_h,
                save_dir=params.save_dir,
            )
    elif params.method == 'anls':
        import anls
        for features, lambda_w, lambda_h in product(params.features,
                                                    params.lambda_w,
                                                    params.lambda_h):
            anls.anls(
                data,
                features,
                use_fcnnls=params.use_fcnnls,
                lambda_w=lambda_w,
                lambda_h=lambda_h,
                min_iter=params.min_iter,
                max_iter=params.max_iter,
                tol1=params.tol1,
                tol2=params.tol2,
                save_dir=params.save_dir,
                )
    elif params.method == 'admm':
        import admm
        for features, rho in product(params.features,
                                     params.rho,
                                     ):
            admm.admm(
                data,
                features,
                rho=rho,
                min_iter=params.min_iter,
                max_iter=params.max_iter,
                tol1=params.tol1,
                tol2=params.tol2,
                save_dir=params.save_dir,
            )
    elif params.method == 'admm_nnls':
        import admm_nnls
        for features, rho, lambda_w, lambda_h in product(params.features,
                                                         params.rho,
                                                         params.lambda_w,
                                                         params.lambda_h):
            admm_nnls.admm_nnls(
                data,
                features,
                rho=rho,
                use_fcnnls=params.use_fcnnls,
                lambda_w=lambda_w,
                lambda_h=lambda_h,
                min_iter=params.min_iter,
                max_iter=params.max_iter,
                tol1=params.tol1,
                tol2=params.tol2,
                save_dir=params.save_dir,
                )
    else:
        raise KeyError('Unknown method type.')
