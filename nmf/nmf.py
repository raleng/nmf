import os
from importlib import import_module

from . import utils


class NMF:
    """ Non-negative matrix factorization using MUR, ANLS, ADMM or AO-ADMM 

    This class creates a NMF instance given some non-negative 2D "data" and computes the factorization
    with "factors" number of factors. The results of the factorization can be accessed as the 
    factors "w" and "h".

    There are four methods available: mur, anls, admm, ao_admm

    This class provides two methods. "factorize" computes the factorization, "save_factorization" saves
    the factorization to a file.

    It is possible to pass a parameter file containing the method parameters as a dictionary. Currently,
    now sanity check exist, so make sure the correct keywords for each method are used.

    ======
    Example usage:
    
        > from nmf import NMF
        > nmf = NMF(data, factors)
        > nmf.factorize(method='mur', **method_params)
        > print(nmf.w, nmf.h)
        
    """

    def __init__(self, data=None, factors=None, saving=True, param_file=None):
        self.data = data
        self.factors = factors
        self.saving = saving

        # parameter file handling
        if param_file is not None:
            try:
                parameters = import_module(param_file)
            except ImportError:
                print('No parameter file found.')
                return
                
            self.method_params = parameters.method_params


    def factorize(self, method='mur', saving=False, **method_params):
        """ Calculate the factorization

        Possible methods: mur, anls, admm, ao_admm

        See documentation for possible method parameters to adjust the algorithms.

        If the "saving" flag is set to True, the results will be saved automatically after
        computation.
        """

        if method == 'mur':
            from . import mur
            self.results = mur.mur(self.data, self.factors, **method_params)

        elif method == 'anls':
            from . import anls
            self.results = anls.anls(self.data, self.factors, **method_params)

        elif method == 'admm':
            from . import admm
            self.results = admm.admm(self.data, self.factors, **method_params)

        elif method == 'ao_admm':
            from . import ao_admm
            self.results = ao_admm.ao_admm(self.data, self.factors, **method_params)

        else:
            raise Exception('Method not known. Choose one from: mur anls admm ao_admm')
    
        print('Factorization done.')
        if saving:
            self.save_factorization() 


    def save_factorization(self, save_dir='./results', save_name=None):
        """ Save factorization to file
        
        Default folder is "./results" and will be created if it doesn't exist. If no "save_name" is given
        a default file name is created based on the parameters of the factorization.
        """
        # TODO: save_name needs to include more parameters and NNDSVD
    
        # create results folder, if not existing
        os.makedirs(save_dir, exist_ok=True)

        # generate standard save_name from parameters
        if save_name is None:
            # prefix 'nmf' and method
            save_name = f'nmf_{self.results.experiment.method}'
            # number of factors
            save_name += f'_{self.factors}'
            # distance type
            save_name += f'_{self.results.experiment.distance_type}'
            # for ADMM, add rho value
            if self.results.experiment.method == 'admm':
                save_name += f'_{self.results.experiment.rho}'

            # regularization value for w
            save_name += f'_{self.results.experiment.lambda_w}'
            # regularization type for w in ADMM / AO-ADMM
            if self.results.experiment.method in {'admm', 'ao_admm'}:
                save_name += f':{self.results.experiment.prox_w}'
            
            # regularization value for h
            save_name += f'_{self.results.experiment.lambda_h}'
            # regularization type for h in ADMM / AO-ADMM
            if self.results.experiment.method in {'admm', 'ao_admm'}:
                save_name += f':{self.results.experiment.prox_h}'
        
            # initialization type
            if self.results.experiment.nndsvd_init[0]:
                save_name += f'_nndsvd{self.results.experiment.nndsvd_init[1][0]}'
            else:
                save_name += '_random'

            # FCNNLS usage in ANLS
            if self.results.experiment.method == 'anls' and self.results.experiment.fcnnls:
                save_name += '_fcnnls'

        # create save string including folder and save
        save_str = os.path.join(save_dir, save_name)
        utils.save_results(save_str,
                     w = self.results.w,
                     h = self.results.h,
                     i = self.results.i,
                     obj_history = self.results.obj_history,
                     experiment = self.results.experiment._asdict())