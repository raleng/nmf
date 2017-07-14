import os
from importlib import import_module

import begin
# noinspection PyUnresolvedReferences
import better_exceptions
import numpy as np
from misc import loadme


def initialize(dims, features):
    return h


def w_update():
    pass


def h_update():
    pass


def normalize(w, h):
    return w, h

def anls():

    h = initialize()

    while not convergence:

        w = w_update()

        h = h_update()

        w, h = normalize(w, h)



@begin.start
def main(param_file='parameters_anls'):
    """ NMF with ANLS """

    try:
        params = import_module(param_file)
    except ImportError:
        print('No parameter file found.')
        return

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

    if data.ndim == 3:
        data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]), order='F')
        print('Data was 3D. Reshaped to 2D.')

    anls(data,
         params.features,
         rho=params.rho,
         max_iter=params.max_iter,
         tol1=params.tol1,
         tol2=params.tol2,
         save_dir=params.save_dir,
         save_file=params.save_file,
         )
