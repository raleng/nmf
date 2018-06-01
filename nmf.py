#!/usr/bin/env python
import begin
from importlib import import_module

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

    # Loading data
    try:
        if params.load_var == 'LOAD_MSOT':
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
    if params.use_fcnnls:
        print('Using FCNNLS.')

    # Method call
    if params.method == 'mur':
        import mur
        mur.mur(
            data,
            params.features,
            distance_type=params.distance_type,
            norm=params.norm,
            max_iter=params.max_iter,
            tol1=params.tol1,
            tol2=params.tol2,
            lambda_w=params.lambda_w,
            lambda_h1=params.lambda_h,
            lambda_h2=params.lambda_h,
            save_dir=params.save_dir,
            save_file=params.save_file,
        )
    elif params.method == 'anls':
        import anls
        anls.anls(
            data,
            params.features,
            use_fcnnls=params.use_fcnnls,
            lambda_w=params.lambda_w,
            lambda_h=params.lambda_h,
            min_iter=params.min_iter,
            max_iter=params.max_iter,
            tol1=params.tol1,
            tol2=params.tol2,
            save_dir=params.save_dir,
            save_file=params.save_file,
            )
    elif params.method == 'admm':
        import admm
        admm.admm(
            data,
            params.features,
            rho=params.rho,
            min_iter=params.min_iter,
            max_iter=params.max_iter,
            tol1=params.tol1,
            tol2=params.tol2,
            save_dir=params.save_dir,
            save_file=params.save_file,
        )
    elif params.method == 'admm_nnls':
        import admm_nnls
        admm_nnls.admm(
            data,
            params.features,
            rho=params.rho,
            use_fcnnls=params.use_fcnnls,
            lambda_w=params.lambda_w,
            lambda_h=params.lambda_h,
            min_iter=params.min_iter,
            max_iter=params.max_iter,
            tol1=params.tol1,
            tol2=params.tol2,
            save_dir=params.save_dir,
            save_file=params.save_file,
            )
    else:
        raise KeyError('Unknown method type.')
